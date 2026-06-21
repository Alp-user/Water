#version 460

#define QUAD 0
#define BILL_QUAD 1
#define NORMAL 2
#define UNPROJECT_QUAD 3
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_offset;

layout(location = 0) uniform mat4 model;
layout(location = 1) uniform mat4 view;
layout(location = 2) uniform mat4 projection;
layout(location = 3) uniform mat3 mnormal;
layout(location = 4) uniform bool instanced;
layout(location = 5) uniform int mode;
layout(location = 6) uniform float radius;
layout(location = 7) uniform mat4 inv_projection;

layout(location = 0) out vec3 wpos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec3 center;

out gl_PerVertex {
    vec4 gl_Position;
};

void main(void) {
    switch (mode) {
        case QUAD:
        {
            vec3 vertices[3] = {
                    vec3(-1.0f, 3.0f, -1.0f),
                    vec3(-1.0f, -1.0f, -1.0f),
                    vec3(3.0f, -1.0f, -1.0f)
                };
            vec2 uvs[3] = {
                    vec2(0.0f, 2.0f),
                    vec2(0.0f, 0.0f),
                    vec2(2.0f, 0.0f)
                };
            wpos = vertices[gl_VertexID];
            uv = uvs[gl_VertexID];
            gl_Position = vec4(vertices[gl_VertexID], 1.0f);
            break;
        }
        case BILL_QUAD:
        {
            vec3 center_view = vec3(view * vec4(in_offset, 1.0f));
            vec3 bottom_left = center_view - vec3(radius, radius, 0.0f);
            center = center_view;
            switch (gl_VertexID) {
                case 5:
                case 0:
                {
                    uv = vec2(0, 1);
                    vec3 top_left = bottom_left + vec3(0.0f, 2.0f * radius, 0.0f);
                    wpos = top_left;
                    gl_Position = projection * vec4(top_left, 1.0f);
                    break;
                }
                case 1:
                {
                    uv = vec2(0, 0);
                    wpos = bottom_left;
                    gl_Position = projection * vec4(bottom_left, 1.0f);
                    break;
                }
                case 3:
                case 2:
                {
                    uv = vec2(1, 0);
                    vec3 bottom_right = bottom_left + vec3(2.0f * radius, 0.0f, 0.0f);
                    wpos = bottom_right;
                    gl_Position = projection * vec4(bottom_right, 1.0f);
                    break;
                }
                case 4:
                {
                    uv = vec2(1, 1);
                    vec3 top_right = bottom_left + vec3(2.0f * radius, 2.0f * radius, 0.0f);
                    wpos = top_right;
                    gl_Position = projection * vec4(top_right, 1.0f);
                    break;
                }
            }
            break;
        }
        case NORMAL:
        {
            uv = in_uv;
            normal = mnormal * in_normal;
            wpos = vec3(model * vec4(pos, 1.0f));
            if (instanced) wpos += in_offset;
            gl_Position = projection * view * vec4(wpos, 1.0f);
            break;
        }
        case UNPROJECT_QUAD:
        {
            vec3 vertices[3] = {
                    vec3(-1.0f, 3.0f, -1.0f),
                    vec3(-1.0f, -1.0f, -1.0f),
                    vec3(3.0f, -1.0f, -1.0f)
                };
            vec2 uvs[3] = {
                    vec2(0.0f, 2.0f),
                    vec2(0.0f, 0.0f),
                    vec2(2.0f, 0.0f)
                };
            vec4 view_first = inv_projection * vec4(vertices[gl_VertexID], 1.0f);
            wpos = view_first.xyz / view_first.w;
            wpos = wpos / abs(wpos.z); // z = 1.0f so that now multiplying by depth gives the view location
            uv = uvs[gl_VertexID];
            gl_Position = vec4(vertices[gl_VertexID], 1.0f);
            break;
        }
    }
}
