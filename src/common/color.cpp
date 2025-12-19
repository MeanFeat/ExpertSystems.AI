#include "es_core_pch.h"
#include "color.h"

int Color::ToBit() const {
	return ((r << 16) | ((g << 8) | b));
}

Color Color::Blend(const Color &other, const float percent) const {
	return Color(
		static_cast<unsigned char>(float(r) + (percent * (float(other.r) - float(r)))),
		static_cast<unsigned char>(float(g) + (percent * (float(other.g) - float(g)))),
		static_cast<unsigned char>(float(b) + (percent * (float(other.b) - float(b)))),
		static_cast<unsigned char>(float(a) + (percent * (float(other.a) - float(a))))
	);
}