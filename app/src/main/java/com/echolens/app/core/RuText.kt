package com.echolens.app.core

import java.util.Locale

/** Russian phrase helpers for scene speech and the on-screen track list. */
object RuText {

    fun direction(azDeg: Float): String = when {
        azDeg > -15 && azDeg <= 15 -> "прямо"
        azDeg > 15 && azDeg <= 45 -> "чуть справа"
        azDeg > 45 && azDeg <= 110 -> "справа"
        azDeg > 110 || azDeg <= -110 -> "позади"
        azDeg > -110 && azDeg <= -45 -> "слева"
        else -> "чуть слева"
    }

    fun distance(m: Float): String = when {
        m < 1f -> "ближе метра"
        else -> {
            val n = Math.round(m).coerceAtMost(99)
            String.format(Locale.US, "%d %s", n, metersPlural(n))
        }
    }

    private fun metersPlural(n: Int): String {
        val n10 = n % 10
        val n100 = n % 100
        return when {
            n10 == 1 && n100 != 11 -> "метр"
            n10 in 2..4 && n100 !in 12..14 -> "метра"
            else -> "метров"
        }
    }
}
