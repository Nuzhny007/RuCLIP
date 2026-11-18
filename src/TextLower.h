#pragma once

#include <cstdint>
#include <iterator>
#include <string>

#include "utf8.h"

/// Понижает регистр UTF-8-строки для диапазонов, которые реально встречаются в
/// русских подписях: ASCII (A-Z), Latin-1 (À-Þ) и кириллица (А-Я, Ё).
///
/// Байты, не образующие корректную UTF-8-последовательность, копируются без
/// изменений: так текст, пришедший в ANSI-кодировке, хотя бы не портится.
/// Полная поддержка Unicode и нормализация кодировки argv на Windows — отдельная
/// задача, см. roadmap 4.6.
inline uint32_t ToLowerCodepoint(uint32_t codepoint)
{
	if (codepoint >= 0x41 && codepoint <= 0x5A)                             // ASCII A-Z
		return codepoint + 0x20;
	if (codepoint >= 0x00C0 && codepoint <= 0x00DE && codepoint != 0x00D7) // Latin-1 À-Þ, кроме ×
		return codepoint + 0x20;
	if (codepoint >= 0x0400 && codepoint <= 0x040F)                        // Кириллица Ѐ-Џ (включая Ё U+0401)
		return codepoint + 0x50;
	if (codepoint >= 0x0410 && codepoint <= 0x042F)                        // Кириллица А-Я
		return codepoint + 0x20;
	return codepoint;
}

inline std::string ToLowerUtf8(const std::string& text)
{
	std::string lowered;
	lowered.reserve(text.size());

	const char* cursor = text.data();
	const char* const end = cursor + text.size();
	while (cursor < end)
	{
		uint64_t length = 0;
		const uint32_t codepoint = vkcom::chars_to_utf8(cursor, static_cast<uint64_t>(end - cursor), &length);
		if (codepoint == vkcom::INVALID_UNICODE)
		{
			// Не UTF-8: оставляем байт без изменений и идём дальше.
			lowered.push_back(*cursor);
			++cursor;
			continue;
		}
		vkcom::utf8_to_chars(ToLowerCodepoint(codepoint), std::back_inserter(lowered));
		cursor += length;
	}
	return lowered;
}
