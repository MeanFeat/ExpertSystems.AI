#pragma once
#ifndef COLOR_H
#define COLOR_H

/**
 * RGBA color class for graphics operations
 */
class Color {
public:
	Color(int inR, int inG, int inB, int inA) : r(inR), g(inG), b(inB), a(inA) {}
	
	int r, g, b, a;
	
	/**
	 * Convert color to 32-bit integer representation
	 * @return RGB value packed into integer
	 */
	int ToBit() const;
	
	/**
	 * Blend this color with another color
	 * @param other Color to blend with
	 * @param percent Blend percentage (0.0 to 1.0)
	 * @return Blended color
	 */
	Color Blend(const Color &other, float percent) const;
	
	Color operator-(const Color &other) const {
		return Color(r - other.r, g - other.g, b - other.b, a - other.a);
	}
	
	Color operator*(const float scalar) const {
		return Color(int(r * scalar), int(g * scalar), int(b * scalar), int(a * scalar));
	}
};

#endif