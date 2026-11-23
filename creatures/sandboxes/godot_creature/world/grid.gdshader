shader_type spatial;
// World-space grid: thin line every cell (1 m), bold line every major cell (5 m), light checker tint.
uniform vec3 base_color : source_color = vec3(0.33, 0.40, 0.30);
uniform vec3 alt_color : source_color = vec3(0.37, 0.44, 0.33);
uniform vec3 line_color : source_color = vec3(0.16, 0.20, 0.16);
uniform float cell = 1.0;
uniform float major = 5.0;
varying vec3 wpos;

void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

float grid_line(vec2 p, float size, float width) {
	vec2 c = p / size;
	vec2 g = abs(fract(c - 0.5) - 0.5) / (fwidth(c) * width);
	return 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
}

void fragment() {
	vec2 p = wpos.xz;
	vec2 id = floor(p / cell);
	float checker = mod(id.x + id.y, 2.0);
	float fade = clamp(1.0 - length(fwidth(p / cell)) * 1.5, 0.0, 1.0);   // no moire far away
	vec3 col = mix(base_color, mix(base_color, alt_color, checker), fade);
	float l = max(grid_line(p, cell, 1.2) * 0.6 * fade, grid_line(p, cell * major, 2.0));
	ALBEDO = mix(col, line_color, l);
	ROUGHNESS = 0.95;
}
