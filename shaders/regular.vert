#version 460

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) uniform mat4 model;
layout(location = 1) uniform mat4 view;
layout(location = 2) uniform mat4 projection;
layout(location = 3) uniform mat3 mnormal;

layout(location = 0) out vec3 wpos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;


out gl_PerVertex{
    vec4 gl_Position;
};

void main(void){
    uv = in_uv;
    normal = mnormal * in_normal;
    wpos = vec3(model * vec4(pos, 1.0f));
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}
