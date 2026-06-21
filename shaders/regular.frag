#version 460

#define SHADING_MODE 0
#define BLURX_MODE 1
#define BLURY_MODE 2
#define SPHERE_MODE 3
#define WFINAL_MODE 4

layout(location = 0) in vec3 wpos; // view pos in bill_quad case
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 center;

layout(location = 0) uniform vec3 cpos;
layout(location = 1) uniform int mode;
layout(location = 2) uniform vec2 hv_inc;
layout(location = 3) uniform float radius;
layout(location = 4) uniform mat4 projection;
layout(location = 5) uniform float projection_scale;
layout(location = 6) uniform float world_blur_radius;
layout(location = 7) uniform vec3 light_direction;
layout(location = 8) uniform mat4 inv_view;

layout(binding = 0) uniform sampler2D offtex;

layout(location = 0) out vec4 fbo_color;

vec4 diffuse_color(vec3 surface_normal, vec3 light_direction, vec4 texture_color, vec4 light_color) {
    float n_dot_l = dot(surface_normal, light_direction);
    return texture_color * light_color * max(0, n_dot_l); // NOTE: Add multiple light sources
}

vec4 specular_color(vec3 surface_normal, vec3 light_direction, vec3 cam_direction, float texture_specularity, vec4 light_color, float exponent) {
    vec3 half_vec = normalize(light_direction + cam_direction);
    float n_dot_h = dot(surface_normal, half_vec);
    float n_dot_l = dot(surface_normal, light_direction);
    if (n_dot_l < 0.0f) return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return texture_specularity * light_color * pow(max(0, n_dot_h), exponent);
}

vec4 ambient_color(float ambient_coefficient) {
    vec4 ambient_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    return ambient_color * ambient_coefficient;
}

void main(void) {
    switch (mode) {
        case SHADING_MODE:
        {
            vec4 texture_color = vec4(0.0f, 0.0f, 1.0f, 1.0f);
            vec4 light_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
            vec3 light_dir = vec3(1.0f, 1.0f, 0.0f);
            vec3 cam_dir = normalize(cpos - wpos);
            vec3 surface_normal = normalize(normal);

            vec4 diffuse = diffuse_color(surface_normal, light_dir, texture_color, light_color);
            vec4 specular = specular_color(surface_normal, light_dir, cam_dir, 1.0f, light_color, 50.0f);
            vec4 ambient = ambient_color(0.5f);

            fbo_color = diffuse + specular + ambient;
            break;
        }
        case BLURY_MODE: // NOTE: explain you have two passes for blur for performance
        case BLURX_MODE:
        {
            float blur_scale = 1.0f;
            float blur_depth_falloff = 1.0f;
            float cdepth = -texture(offtex, uv).z; // Camera looks at -z so negate
            if (cdepth < 0.01f) {
                fbo_color = vec4(0.0f);
                break;
            }
            // if(cdepth == 1.0f) {fbo_color = vec4(1.0f); break;}
            float pixel_radius = world_blur_radius * projection_scale / cdepth;
            pixel_radius = min(pixel_radius, 50.0f);
            float blurred_depth = 0.0f;
            float total_weight = 0.0f;
            for (float i = -pixel_radius; i <= pixel_radius; i++) {
                vec2 nuv;
                if (mode == BLURX_MODE) nuv = uv + vec2(i * hv_inc.x, 0.0f);
                else nuv = uv + vec2(0.0f, i * hv_inc.y);
                float sampled_depth = -texture(offtex, nuv).z;
                if (sampled_depth != sampled_depth || sampled_depth < 0.1f) continue;
                float spatial_weight = exp(-(i * i * blur_scale));
                float content = (cdepth - sampled_depth) * blur_depth_falloff;
                float edge_weight = exp(-(content * content));
                float sum_weight = edge_weight * spatial_weight;
                total_weight += sum_weight;
                blurred_depth += sum_weight * sampled_depth;
            }
            if (total_weight < 0.01f) {
                fbo_color = vec4(0.0f, 0.0f, cdepth, 1.0f); // Neighbors did not contribute
                break;
            }
            blurred_depth /= total_weight;
            fbo_color = vec4(0.0f, 0.0f, -blurred_depth, 1.0f); // Negate to go back to -z

            break;
        }
        case SPHERE_MODE:
        {
            vec3 dir = wpos - center;
            if (length(dir.xy) > radius) discard;
            float depth = sqrt(radius * radius - dir.x * dir.x - dir.y * dir.y) + center.z;
            vec4 view_space_pos = vec4(wpos.x, wpos.y, depth, 1.0f);
            vec4 clip_space_pos = projection * view_space_pos;
            gl_FragDepth = clip_space_pos.z / clip_space_pos.w;
            fbo_color = vec4(view_space_pos.xyz, 1.0f);
            break;
        }
        case WFINAL_MODE:
        {
            float depth_val = texture(offtex, uv).z;
            if (depth_val == 1.0f) discard; // Background
            vec3 pos = wpos * depth_val;

            float right_depth = texture(offtex, uv + vec2(hv_inc.x, 0.0f)).z;
            float top_depth = texture(offtex, uv + vec2(0.0f, hv_inc.y)).z;
            float left_depth = texture(offtex, uv + vec2(-hv_inc.x, 0.0f)).z;
            float bottom_depth = texture(offtex, uv + vec2(0.0f, -hv_inc.y)).z;

            vec3 wpos_dx = dFdx(wpos);
            vec3 wpos_dy = dFdy(wpos);

            vec3 dxr = (wpos + wpos_dx) * right_depth;
            vec3 dyt = (wpos + wpos_dy) * top_depth;
            vec3 dxl = (wpos - wpos_dx) * left_depth;
            vec3 dyb = (wpos - wpos_dy) * bottom_depth;

            vec3 dx = dxr - pos;
            vec3 dx2 = pos - dxl;

            if (abs(dx.z) > abs(dx2.z)) {
                dx = dx2;
            }

            vec3 dy = dyt - pos;
            vec3 dy2 = pos - dyb;

            if (abs(dy.z) > abs(dy2.z)) {
                dy = dy2;
            }
            vec3 surface_normal = normalize(cross(dx, dy));

            // Works because 3x3 portion(rotation u|v|w) orthogonal and inverse transpose is same as itself
            surface_normal = normalize(mat3(inv_view) * surface_normal);
            vec3 world_pos = (inv_view * vec4(pos, 1.0f)).xyz;
            vec4 texture_color = vec4(0.7f, 0.25f, 0.95f, 0.0f);
            vec4 light_color = vec4(0.1f, 0.1f, 0.1f, 0.0f);
            vec3 cam_dir = normalize(cpos - world_pos);
            vec3 light_dir = vec3(0.0f, 1.0f, 0.0f);
            vec3 light_dirs[9] = vec3[9](
                    vec3(1.0f, 1.0f, 0.0f),
                    vec3(1.0f, -1.0f, 0.0f),
                    vec3(-1.0f, 1.0f, 0.0f),
                    vec3(-1.0f, -1.0f, 0.0f),
                    vec3(0.0f, 1.0f, 1.0f),
                    vec3(0.0f, -1.0f, 1.0f),
                    vec3(0.0f, 1.0f, -1.0f),
                    vec3(0.0f, -1.0f, -1.0f),
                    vec3(0.0f, 1.0f, 0.0f)
                );

            fbo_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = 0; i < 9; i++) {
                vec4 diffuse = diffuse_color(surface_normal, light_dirs[i], texture_color, light_color);
                vec4 specular = specular_color(surface_normal, light_dirs[i], cam_dir, 0.1f, light_color, 50.0f);
                fbo_color += diffuse + specular;
            }

            break;
        }
        default:
        {
            fbo_color = vec4(0.0f, 0.0f, 1.0f, 1.0f);
            break;
        }
    }
}
