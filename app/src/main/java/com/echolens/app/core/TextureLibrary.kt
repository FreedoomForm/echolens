package com.echolens.app.core

/**
 * Maps COCO object classes to their sound-producing material,
 * Russian name, typical height and navigation importance.
 *
 * The idea: every recognized object starts "sounding" like its material —
 * a chair knocks with wood, a car rumbles, a glass clinks. The listener
 * learns to recognize what surrounds them without any speech.
 */
object TextureLibrary {

    private val defaultSpec = MaterialSpec(Material.NEUTRAL, "объект", 0.4f, 0.35f, 1.5f)

    private val map = HashMap<String, MaterialSpec>().apply {
        fun entry(label: String, material: Material, ru: String, h: Float, imp: Float, ref: Float) {
            put(label, MaterialSpec(material, ru, h, imp, ref))
        }

        // People
        entry("person", Material.PERSON, "человек", 1.65f, 1.0f, 2.5f)

        // Vehicles — engines rumble
        entry("bicycle", Material.METAL, "велосипед", 1.0f, 0.55f, 2.0f)
        entry("car", Material.ENGINE, "машина", 1.5f, 1.0f, 5.0f)
        entry("motorcycle", Material.ENGINE, "мотоцикл", 1.4f, 0.9f, 4.0f)
        entry("airplane", Material.ENGINE, "самолёт", 3.0f, 0.8f, 8.0f)
        entry("bus", Material.ENGINE, "автобус", 3.2f, 1.0f, 6.0f)
        entry("train", Material.ENGINE, "поезд", 3.5f, 1.0f, 6.0f)
        entry("truck", Material.ENGINE, "грузовик", 3.0f, 1.0f, 5.0f)
        entry("boat", Material.WOOD, "лодка", 1.5f, 0.6f, 4.0f)

        // Street furniture
        entry("traffic light", Material.ELECTRONICS, "светофор", 0.8f, 0.7f, 3.0f)
        entry("fire hydrant", Material.METAL, "гидрант", 0.75f, 0.3f, 2.0f)
        entry("stop sign", Material.METAL, "знак «стоп»", 0.75f, 0.4f, 2.0f)
        entry("parking meter", Material.METAL, "паркомат", 1.2f, 0.3f, 2.0f)
        entry("bench", Material.WOOD, "скамейка", 0.85f, 0.5f, 2.5f)

        // Animals
        entry("bird", Material.BIRD, "птица", 0.18f, 0.7f, 2.0f)
        entry("cat", Material.ANIMAL, "кошка", 0.3f, 0.8f, 2.0f)
        entry("dog", Material.ANIMAL, "собака", 0.55f, 0.85f, 2.0f)
        entry("horse", Material.ANIMAL, "лошадь", 1.6f, 0.7f, 3.0f)
        entry("sheep", Material.ANIMAL, "овца", 0.8f, 0.6f, 3.0f)
        entry("cow", Material.ANIMAL, "корова", 1.4f, 0.6f, 3.0f)
        entry("elephant", Material.ANIMAL, "слон", 2.8f, 0.6f, 5.0f)
        entry("bear", Material.ANIMAL, "медведь", 1.2f, 0.7f, 4.0f)
        entry("zebra", Material.ANIMAL, "зебра", 1.4f, 0.6f, 4.0f)
        entry("giraffe", Material.ANIMAL, "жираф", 4.5f, 0.6f, 6.0f)

        // Bags and fabrics
        entry("backpack", Material.FABRIC, "рюкзак", 0.45f, 0.4f, 1.5f)
        entry("umbrella", Material.FABRIC, "зонт", 0.8f, 0.4f, 1.5f)
        entry("handbag", Material.FABRIC, "сумка", 0.3f, 0.4f, 1.5f)
        entry("tie", Material.FABRIC, "галстук", 0.4f, 0.3f, 1.2f)
        entry("suitcase", Material.FABRIC, "чемодан", 0.6f, 0.5f, 2.0f)

        // Sports gear
        entry("frisbee", Material.NEUTRAL, "фрисби", 0.05f, 0.3f, 1.0f)
        entry("skis", Material.WOOD, "лыжи", 1.6f, 0.4f, 3.0f)
        entry("snowboard", Material.WOOD, "сноуборд", 1.5f, 0.4f, 3.0f)
        entry("sports ball", Material.NEUTRAL, "мяч", 0.22f, 0.5f, 1.5f)
        entry("kite", Material.FABRIC, "воздушный змей", 0.6f, 0.3f, 2.0f)
        entry("baseball bat", Material.WOOD, "бейсбольная бита", 0.85f, 0.3f, 1.5f)
        entry("baseball glove", Material.FABRIC, "бейсбольная перчатка", 0.3f, 0.3f, 1.2f)
        entry("skateboard", Material.WOOD, "скейтборд", 0.25f, 0.4f, 1.5f)
        entry("surfboard", Material.WOOD, "серфборд", 1.8f, 0.4f, 3.0f)
        entry("tennis racket", Material.METAL, "ракетка", 0.6f, 0.3f, 1.5f)

        // Tableware
        entry("bottle", Material.GLASS, "бутылка", 0.26f, 0.45f, 1.2f)
        entry("wine glass", Material.GLASS, "бокал", 0.2f, 0.45f, 1.0f)
        entry("cup", Material.CERAMIC, "чашка", 0.1f, 0.45f, 1.0f)
        entry("fork", Material.METAL, "вилка", 0.18f, 0.25f, 0.8f)
        entry("knife", Material.METAL, "нож", 0.22f, 0.3f, 0.8f)
        entry("spoon", Material.METAL, "ложка", 0.18f, 0.25f, 0.8f)
        entry("bowl", Material.CERAMIC, "миска", 0.12f, 0.4f, 1.0f)

        // Food
        entry("banana", Material.FOOD, "банан", 0.1f, 0.3f, 1.0f)
        entry("apple", Material.FOOD, "яблоко", 0.08f, 0.35f, 1.0f)
        entry("sandwich", Material.FOOD, "сэндвич", 0.08f, 0.3f, 1.0f)
        entry("orange", Material.FOOD, "апельсин", 0.08f, 0.3f, 1.0f)
        entry("broccoli", Material.FOOD, "брокколи", 0.15f, 0.25f, 1.0f)
        entry("carrot", Material.FOOD, "морковь", 0.18f, 0.25f, 1.0f)
        entry("hot dog", Material.FOOD, "хот-дог", 0.08f, 0.3f, 1.0f)
        entry("pizza", Material.FOOD, "пицца", 0.1f, 0.3f, 1.0f)
        entry("donut", Material.FOOD, "пончик", 0.08f, 0.3f, 1.0f)
        entry("cake", Material.FOOD, "торт", 0.15f, 0.35f, 1.0f)

        // Furniture
        entry("chair", Material.WOOD, "стул", 0.85f, 0.6f, 2.0f)
        entry("couch", Material.FABRIC, "диван", 0.85f, 0.6f, 2.5f)
        entry("potted plant", Material.PLANT, "растение в горшке", 0.6f, 0.5f, 1.5f)
        entry("bed", Material.FABRIC, "кровать", 0.9f, 0.6f, 2.5f)
        entry("dining table", Material.WOOD, "стол", 0.75f, 0.6f, 2.5f)
        entry("toilet", Material.CERAMIC, "унитаз", 0.75f, 0.5f, 2.0f)

        // Electronics
        entry("tv", Material.ELECTRONICS, "телевизор", 0.6f, 0.6f, 3.0f)
        entry("laptop", Material.ELECTRONICS, "ноутбук", 0.24f, 0.5f, 1.5f)
        entry("mouse", Material.ELECTRONICS, "мышь", 0.05f, 0.3f, 1.0f)
        entry("remote", Material.ELECTRONICS, "пульт", 0.16f, 0.35f, 1.0f)
        entry("keyboard", Material.ELECTRONICS, "клавиатура", 0.15f, 0.4f, 1.0f)
        entry("cell phone", Material.ELECTRONICS, "телефон", 0.15f, 0.5f, 1.0f)
        entry("microwave", Material.ELECTRONICS, "микроволновка", 0.3f, 0.4f, 1.5f)
        entry("oven", Material.ELECTRONICS, "духовка", 0.7f, 0.4f, 2.0f)
        entry("toaster", Material.ELECTRONICS, "тостер", 0.2f, 0.3f, 1.2f)
        entry("sink", Material.METAL, "раковина", 0.25f, 0.4f, 1.5f)
        entry("refrigerator", Material.ELECTRONICS, "холодильник", 1.7f, 0.5f, 2.5f)

        // Misc household
        entry("book", Material.PAPER, "книга", 0.24f, 0.4f, 1.2f)
        entry("clock", Material.ELECTRONICS, "часы", 0.3f, 0.4f, 1.5f)
        entry("vase", Material.CERAMIC, "ваза", 0.25f, 0.4f, 1.2f)
        entry("scissors", Material.METAL, "ножницы", 0.15f, 0.3f, 1.0f)
        entry("teddy bear", Material.FABRIC, "игрушечный медведь", 0.35f, 0.4f, 1.2f)
        entry("hair drier", Material.ELECTRONICS, "фен", 0.2f, 0.3f, 1.0f)
        entry("toothbrush", Material.NEUTRAL, "зубная щётка", 0.18f, 0.25f, 1.0f)
    }

    fun spec(label: String): MaterialSpec = map[label] ?: defaultSpec
}
