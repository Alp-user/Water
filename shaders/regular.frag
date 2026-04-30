#version 460

layout(location = 0) in vec3 wpos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) uniform vec3 cpos;

layout(location = 0) out vec4 fbo_color;

vec4 diffuse_color(vec3 surface_normal, vec3 light_direction, vec4 texture_color, vec4 light_color){
    float n_dot_l = dot(surface_normal, light_direction);
    return texture_color * light_color * max(0, n_dot_l);
}

vec4 specular_color(vec3 surface_normal, vec3 light_direction, vec3 cam_direction, float texture_specularity, vec4 light_color, float exponent ){
    vec3 half_vec = normalize(light_direction + cam_direction);
    float n_dot_h = dot(surface_normal, half_vec);
    float n_dot_l = dot(surface_normal, light_direction);
    if(n_dot_l < 0.0f) return vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return texture_specularity * light_color * pow(max(0, n_dot_h), exponent);
}

vec4 ambient_color(float ambient_coefficient){
    vec4 ambient_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    return ambient_color * ambient_coefficient;
}


void main(void){
    vec4 texture_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    vec4 light_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    vec3 light_dir = vec3(1.0f, 1.0f, 0.0f);
    vec3 cam_dir = normalize(cpos - wpos);
    vec3 surface_normal = normalize(normal);

    vec4 diffuse = diffuse_color(surface_normal, light_dir, texture_color, light_color);
    vec4 specular = specular_color(surface_normal, light_dir, cam_dir, 1.0f, light_color, 50.0f);
    vec4 ambient = ambient_color(0.5f);

    fbo_color = diffuse + specular + ambient;
}
