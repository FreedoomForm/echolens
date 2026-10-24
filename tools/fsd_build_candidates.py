#!/usr/bin/env python3
"""Build candidate clip lists for EchoLens class/material sound packs from FSD50K.

Inputs (in /home/z/my-project/data):
  dev.csv, eval.csv               - FSD50K labels (fname,labels,mids[,split])
  eval_clips_info.json            - per-clip title/tags/license/uploader (eval)
  dev_clips_info.json             - same for dev
  pp_pnp.json                     - presence/prominence ratings per clip per mid
  labeltable.json                 - EchoLens LabelTable EN/RU/MAT arrays (426)

Output: candidates.json { target: {"kind": "class"|"material", "material": str,
  "clips": [{"id","lic","promo","uploader","title"}...] } }
"""
import csv, json, os
from collections import defaultdict

DATA = "/home/z/my-project/data"
OUT = os.path.join(DATA, "candidates.json")
MAX_PER_TARGET = 26          # candidates to download per target
MAX_CC0, MAX_CCBY = 18, 10   # priority split

# ---------------------------------------------------------------- mapping
L = lambda *x: list(x)
MAPPING = {
    # ---------------- COCO-80 ----------------
    "person":        (L("Speech", "Conversation", "Chatter", "Human_voice"), None),
    "bicycle":       (L("Bicycle", "Bicycle_bell"), None),
    "car":           (L("Car", "Car_passing_by", "Idling", "Accelerating_and_revving_and_vroom"), None),
    "motorcycle":    (L("Motorcycle", "Accelerating_and_revving_and_vroom"), None),
    "airplane":      (L("Aircraft", "Fixed-wing_aircraft_and_airplane"), None),
    "bus":           (L("Bus"), None),
    "train":         (L("Train", "Rail_transport", "Subway_and_metro_and_underground"), None),
    "truck":         (L("Truck"), None),
    "boat":          (L("Boat_and_Water_vehicle"), None),
    "bench":         (L("Wood"), None),
    "bird":          (L("Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Bird", "Gull_and_seagull", "Crow"), None),
    "cat":           (L("Cat", "Meow", "Purr"), None),
    "dog":           (L("Dog", "Bark", "Growling"), None),
    "horse":         (L("Livestock_and_farm_animals_and_working_animals"), L("horse", "neigh", "whinny")),
    "sheep":         (L("Livestock_and_farm_animals_and_working_animals"), L("sheep", "lamb", "bleat")),
    "cow":           (L("Livestock_and_farm_animals_and_working_animals"), L("cow", "moo", "cattle")),
    "elephant":      (L("Wild_animals"), L("elephant")),
    "bear":          (L("Wild_animals", "Growling"), L("bear")),
    "zebra":         (L("Wild_animals"), L("zebra")),
    "giraffe":       (L("Wild_animals"), L("giraffe")),
    "backpack":      (L("Zipper_(clothing)", "Crumpling_and_crinkling"), None),
    "handbag":       (L("Zipper_(clothing)"), None),
    "suitcase":      (L("Zipper_(clothing)", "Mechanisms"), L("suitcase", "luggage")),
    "sports_ball":   (L("Thump_and_thud", "Hands"), L("ball", "bounce", "dribble")),
    "skateboard":    (L("Skateboard"), None),
    "bottle":        (L("Glass", "Chink_and_clink", "Fill_(with_liquid)", "Pour"), L("bottle")),
    "wine glass":    (L("Glass", "Chink_and_clink", "Shatter"), L("wine glass", "wineglass", "glass clink")),
    "cup":           (L("Dishes_and_pots_and_pans", "Chink_and_clink"), L("cup", "mug", "ceramic")),
    "fork":          (L("Cutlery_and_silverware"), L("fork")),
    "knife":         (L("Cutlery_and_silverware"), L("knife", "chopping")),
    "spoon":         (L("Cutlery_and_silverware"), L("spoon")),
    "bowl":          (L("Dishes_and_pots_and_pans"), L("bowl")),
    "banana":        (L("Chewing_and_mastication"), L("banana", "bite", "crunch")),
    "apple":         (L("Chewing_and_mastication"), L("apple", "bite", "crunch")),
    "sandwich":      (L("Chewing_and_mastication"), L("sandwich", "eating")),
    "orange":        (L("Chewing_and_mastication"), L("orange")),
    "carrot":        (L("Chewing_and_mastication"), L("carrot", "crunch", "vegetable")),
    "pizza":         (L("Chewing_and_mastication"), L("pizza", "eating")),
    "hot dog":       (L("Chewing_and_mastication"), L("hot dog", "eating")),
    "donut":         (L("Chewing_and_mastication"), L("donut", "eating")),
    "cake":          (L("Chewing_and_mastication"), L("cake", "eating")),
    "chair":         (L("Wood", "Squeak"), L("chair", "creak")),
    "dining table":  (L("Dishes_and_pots_and_pans", "Cutlery_and_silverware"), None),
    "toilet":        (L("Toilet_flush"), None),
    "tv":            (L("Buzz"), L("tv", "television", "static")),
    "laptop":        (L("Computer_keyboard", "Typing"), L("laptop", "typing")),
    "mouse":         (L("Typing", "Mechanisms"), L("mouse click", "computer mouse", "click")),
    "remote":        (L("Mechanisms", "Typing"), L("remote", "button", "clicker")),
    "keyboard":      (L("Computer_keyboard", "Typing", "Typewriter"), None),
    "cell phone":    (L("Ringtone", "Telephone", "Buzz"), L("phone", "ring", "vibrat")),
    "microwave":     (L("Microwave_oven", "Bell", "Buzz"), None),
    "sink":          (L("Water_tap_and_faucet", "Sink_(filling_or_washing)", "Drip", "Fill_(with_liquid)"), None),
    "refrigerator":  (L("Buzz"), L("fridge", "refrigerator", "hum", "freezer")),
    "book":          (L("Writing", "Crumpling_and_crinkling"), L("book", "page", "paper")),
    "clock":         (L("Clock", "Tick", "Tick-tock", "Alarm"), None),
    "vase":          (L("Glass", "Chink_and_clink"), L("vase", "ceramic", "porcelain")),
    "scissors":      (L("Scissors"), None),
    "hair drier":    (L("Mechanical_fan", "Power_tool"), L("hair dryer", "hairdryer", "blow dryer")),
    "potted plant":  (L("Wind"), L("leaves", "foliage", "rustling", "tree")),
    # ---------------- extended EchoLens vocabulary ----------------
    "wardrobe":      (L("Cupboard_open_or_close", "Wood"), None),
    "cabinet":       (L("Cupboard_open_or_close", "Drawer_open_or_close"), None),
    "kitchen cabinet": (L("Cupboard_open_or_close"), None),
    "dresser":       (L("Drawer_open_or_close"), None),
    "nightstand":    (L("Drawer_open_or_close", "Wood"), None),
    "bookshelf":     (L("Wood"), None),
    "shelf":         (L("Wood"), None),
    "armchair":      (L("Wood", "Squeak"), L("chair", "creak")),
    "stool":         (L("Wood", "Squeak"), None),
    "coffee table":  (L("Wood"), None),
    "desk":          (L("Wood", "Writing"), None),
    "office chair":  (L("Squeak", "Wood"), L("chair", "swivel", "creak")),
    "door":          (L("Door", "Knock", "Slam", "Squeak", "Sliding_door"), None),
    "window":        (L("Glass", "Squeak", "Knock"), L("window")),
    "mirror":        (L("Glass"), L("mirror")),
    "picture frame": (L("Wood"), None),
    "wall clock":    (L("Clock", "Tick", "Tick-tock"), None),
    "lamp":          (L("Buzz"), L("lamp", "fluorescent")),
    "table lamp":    (L("Buzz"), L("lamp")),
    "floor lamp":    (L("Buzz"), L("lamp")),
    "ceiling light": (L("Buzz"), L("light", "fluorescent")),
    "candle":        (L("Fire"), L("candle", "flame")),
    "trash bin":     (L("Thump_and_thud", "Crumpling_and_crinkling"), L("trash", "bin", "garbage", "lid")),
    "vacuum cleaner": (L("Buzz", "Engine"), L("vacuum")),
    "washing machine": (L("Buzz", "Mechanisms"), L("washing machine", "washer", "laundry")),
    "dishwasher":    (L("Buzz"), L("dishwasher")),
    "fan":           (L("Mechanical_fan"), None),
    "air conditioner": (L("Mechanical_fan", "Buzz"), L("air conditioner", "conditioning")),
    "box":           (L("Wood", "Crumpling_and_crinkling"), L("box", "cardboard")),
    "speaker":       (L("Music"), None),
    "camera":        (L("Camera"), None),
    "video camera":  (L("Camera"), None),
    "computer monitor": (L("Buzz"), L("monitor", "computer")),
    "desktop computer": (L("Buzz"), L("computer", "fan")),
    "printer":       (L("Printer"), None),
    "television":    (L("Buzz"), L("tv", "television")),
    "stove":         (L("Fire", "Frying_(food)"), L("stove", "burner", "gas")),
    "kettle":        (L("Boiling"), L("kettle")),
    "teapot":        (L("Boiling"), L("teapot", "tea")),
    "frying pan":    (L("Frying_(food)", "Dishes_and_pots_and_pans"), None),
    "cooking pot":   (L("Dishes_and_pots_and_pans", "Boiling"), None),
    "plate":         (L("Dishes_and_pots_and_pans"), L("plate", "dish")),
    "saucer":        (L("Dishes_and_pots_and_pans"), L("plate", "saucer")),
    "tray":          (L("Dishes_and_pots_and_pans"), L("tray")),
    "mug":           (L("Dishes_and_pots_and_pans", "Chink_and_clink"), L("mug", "cup")),
    "drinking glass": (L("Glass", "Chink_and_clink"), None),
    "glass jar":     (L("Glass", "Chink_and_clink"), L("jar", "lid")),
    "pitcher":       (L("Fill_(with_liquid)", "Pour"), None),
    "napkin":        (L("Crumpling_and_crinkling", "Tearing"), None),
    "paper towel roll": (L("Tearing", "Crumpling_and_crinkling"), None),
    "strainer":      (L("Dishes_and_pots_and_pans"), L("strainer", "colander")),
    "grater":        (L("Cutlery_and_silverware", "Dishes_and_pots_and_pans"), L("grater")),
    "ladle":         (L("Cutlery_and_silverware", "Dishes_and_pots_and_pans"), L("ladle", "spoon")),
    "spatula":       (L("Cutlery_and_silverware", "Dishes_and_pots_and_pans"), L("spatula")),
    "whisk":         (L("Cutlery_and_silverware", "Dishes_and_pots_and_pans"), L("whisk")),
    "rolling pin":   (L("Wood"), L("rolling pin", "dough")),
    "blender":       (L("Buzz"), L("blender")),
    "coffee maker":  (L("Buzz", "Boiling"), L("coffee maker", "coffee machine", "espresso")),
    "food processor": (L("Buzz"), L("food processor", "blender")),
    "bread":         (L("Chewing_and_mastication"), L("bread", "crust", "crunch")),
    "pastry":        (L("Chewing_and_mastication"), L("pastry", "bite")),
    "croissant":     (L("Chewing_and_mastication"), L("croissant", "crunch")),
    "cookie":        (L("Chewing_and_mastication"), L("cookie", "crunch", "biscuit")),
    "chocolate":     (L("Chewing_and_mastication"), L("chocolate", "bite")),
    "candy":         (L("Chewing_and_mastication"), L("candy", "crunch")),
    "cheese":        (L("Chewing_and_mastication"), L("cheese")),
    "egg":           (L("Crack", "Chink_and_clink"), L("egg")),
    "milk":          (L("Pour", "Liquid", "Gurgling"), L("milk")),
    "juice":         (L("Pour", "Liquid", "Gurgling"), L("juice", "pour")),
    "honey":         (L("Pour", "Liquid"), L("honey")),
    "jam":           (L("Pour", "Liquid"), L("jam", "jar")),
    "ketchup":       (L("Pour", "Liquid", "Trickle_and_dribble"), L("ketchup", "sauce")),
    "sauce":         (L("Pour", "Liquid"), L("sauce")),
    "soup":          (L("Liquid", "Pour", "Gurgling"), L("soup")),
    "pasta":         (L("Chewing_and_mastication"), L("pasta", "noodle")),
    "rice":          (L("Chewing_and_mastication"), L("rice")),
    "cereal":        (L("Chewing_and_mastication", "Crushing"), L("cereal", "crunch")),
    "sausage":       (L("Chewing_and_mastication"), L("sausage")),
    "ham":           (L("Chewing_and_mastication"), L("ham")),
    "bacon":         (L("Chewing_and_mastication", "Frying_(food)"), L("bacon", "sizzle")),
    "meat":          (L("Chewing_and_mastication"), L("meat", "chew")),
    "fish":          (L("Chewing_and_mastication"), L("fish")),
    "shrimp":        (L("Chewing_and_mastication"), L("shrimp")),
    "sushi":         (L("Chewing_and_mastication"), L("sushi")),
    "burger":        (L("Chewing_and_mastication"), L("burger")),
    "french fries":  (L("Chewing_and_mastication", "Crushing"), L("fries", "crunch")),
    "pancakes":      (L("Chewing_and_mastication"), L("pancake")),
    "pie":           (L("Chewing_and_mastication"), L("pie")),
    "ice cream":     (L("Chewing_and_mastication"), L("ice cream", "lick")),
    "watermelon":    (L("Chewing_and_mastication"), L("watermelon", "bite")),
    "grape":         (L("Chewing_and_mastication"), L("grape")),
    "strawberry":    (L("Chewing_and_mastication"), L("strawberry")),
    "tomato":        (L("Chewing_and_mastication"), L("tomato", "bite")),
    "cucumber":      (L("Chewing_and_mastication"), L("cucumber", "crunch")),
    "potato":        (L("Chewing_and_mastication"), L("potato", "crunch")),
    "onion":         (L("Chewing_and_mastication"), L("onion", "chop")),
    "garlic":        (L("Chewing_and_mastication"), L("garlic", "crunch")),
    "bell pepper":   (L("Chewing_and_mastication"), L("pepper", "crunch")),
    "lemon":         (L("Chewing_and_mastication"), L("lemon")),
    "peach":         (L("Chewing_and_mastication"), L("peach", "bite")),
    "pear":          (L("Chewing_and_mastication"), L("pear", "bite")),
    "pineapple":     (L("Chewing_and_mastication", "Crack"), L("pineapple")),
    "mango":         (L("Chewing_and_mastication"), L("mango", "bite")),
    "cabbage":       (L("Chewing_and_mastication"), L("cabbage", "lettuce")),
    "lettuce":       (L("Chewing_and_mastication"), L("lettuce", "crunch")),
    "corn":          (L("Chewing_and_mastication"), L("corn")),
    "mushroom":      (L("Chewing_and_mastication"), L("mushroom")),
    "avocado":       (L("Chewing_and_mastication"), L("avocado")),
    "pumpkin":       (L("Chewing_and_mastication", "Crack"), L("pumpkin")),
    "eggplant":      (L("Chewing_and_mastication"), L("eggplant")),
    "nuts":          (L("Chewing_and_mastication", "Crushing", "Crack"), L("nut", "shell", "crack")),
    "bag":           (L("Crumpling_and_crinkling", "Zipper_(clothing)"), None),
    "plastic bag":   (L("Crumpling_and_crinkling"), L("plastic bag", "plastic")),
    "paper bag":     (L("Crumpling_and_crinkling", "Tearing"), L("paper bag", "paper")),
    "shoes":         (L("Walk_and_footsteps"), L("shoes", "walking", "footstep")),
    "sneaker":       (L("Walk_and_footsteps"), L("sneaker", "footstep")),
    "boot":          (L("Walk_and_footsteps"), L("boot", "footstep")),
    "slipper":       (L("Walk_and_footsteps"), L("slipper", "footstep")),
    "sandal":        (L("Walk_and_footsteps"), L("sandal", "footstep")),
    "glasses":       (L("Chink_and_clink"), L("glasses", "eyeglasses")),
    "sunglasses":    (L("Chink_and_clink"), L("sunglasses", "glasses")),
    "wristwatch":    (L("Tick", "Mechanisms"), L("watch")),
    "key":           (L("Keys_jangling"), None),
    "lighter":       (L("Fire"), L("lighter")),
    "matches":       (L("Fire"), L("match", "strike")),
    "notebook":      (L("Writing"), L("notebook", "writing", "pen")),
    "pen":           (L("Writing"), None),
    "pencil":        (L("Writing"), None),
    "marker":        (L("Writing"), None),
    "stapler":       (L("Mechanisms"), L("stapler", "staple")),
    "glue":          (L("Packing_tape_and_duct_tape"), None),
    "adhesive tape": (L("Packing_tape_and_duct_tape"), None),
    "folder":        (L("Crumpling_and_crinkling"), L("folder", "paper")),
    "envelope":      (L("Tearing", "Crumpling_and_crinkling"), L("envelope", "paper")),
    "document":      (L("Crumpling_and_crinkling", "Writing"), L("paper", "document")),
    "newspaper":     (L("Crumpling_and_crinkling", "Tearing"), L("newspaper", "page")),
    "magazine":      (L("Crumpling_and_crinkling"), L("magazine", "page")),
    "coin":          (L("Coin_(dropping)"), None),
    "bathtub":       (L("Bathtub_(filling_or_washing)", "Water_tap_and_faucet"), None),
    "shower head":   (L("Water_tap_and_faucet", "Trickle_and_dribble"), L("shower")),
    "faucet":        (L("Water_tap_and_faucet", "Drip"), None),
    "toilet paper":  (L("Tearing", "Crumpling_and_crinkling"), None),
    "pill bottle":   (L("Chink_and_clink"), L("pill", "pills")),
    "hammer":        (L("Hammer"), None),
    "screwdriver":   (L("Tools", "Ratchet_and_pawl"), None),
    "wrench":        (L("Tools", "Ratchet_and_pawl"), None),
    "pliers":        (L("Tools"), None),
    "saw":           (L("Sawing"), None),
    "drill":         (L("Drill", "Power_tool"), None),
    "axe":           (L("Wood", "Hammer"), L("axe", "chopping wood", "hatchet")),
    "padlock":       (L("Keys_jangling", "Chink_and_clink"), L("lock", "padlock")),
    "door handle":   (L("Door", "Squeak"), None),
    "chain":         (L("Chink_and_clink", "Keys_jangling"), L("chain")),
    "hose":          (L("Water", "Trickle_and_dribble"), L("hose")),
    "fence":         (L("Wood", "Squeak"), L("fence", "gate")),
    "gate":          (L("Door", "Squeak"), L("gate")),
    "stairs":        (L("Walk_and_footsteps"), L("stairs", "staircase")),
    "escalator":     (L("Mechanisms", "Walk_and_footsteps"), L("escalator")),
    "elevator":      (L("Mechanisms", "Bell"), L("elevator", "lift")),
    "tree":          (L("Wind"), L("leaves", "tree", "foliage", "rustling")),
    "bush":          (L("Wind"), L("leaves", "bush", "rustling")),
    "flower":        (L("Wind"), L("leaves", "flower", "rustling")),
    "fountain":      (L("Water", "Trickle_and_dribble", "Splash_and_splatter"), L("fountain")),
    "swing":         (L("Squeak"), L("swing", "creak")),
    "taxi":          (L("Car", "Car_passing_by", "Vehicle_horn_and_car_horn_and_honking"), L("taxi")),
    "van":           (L("Car", "Motor_vehicle_(road)"), L("van")),
    "tractor":       (L("Engine"), L("tractor")),
    "excavator":     (L("Engine"), L("excavator", "digger")),
    "crane":         (L("Engine"), L("crane")),
    "forklift":      (L("Engine"), L("forklift")),
    "ambulance":     (L("Siren"), L("ambulance")),
    "police car":    (L("Siren"), L("police")),
    "fire truck":    (L("Siren"), L("fire truck", "firetruck")),
    "kick scooter":  (L("Bicycle", "Skateboard"), L("scooter")),
    "scooter":       (L("Bicycle", "Skateboard"), L("scooter")),
    "puddle":        (L("Splash_and_splatter", "Drip"), L("puddle", "splash")),
    "fire":          (L("Fire", "Crackle"), None),
    "pigeon":        (L("Bird_vocalization_and_bird_call_and_bird_song", "Bird"), L("pigeon", "coo")),
    "sparrow":       (L("Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet"), L("sparrow")),
    "crow":          (L("Crow", "Bird_vocalization_and_bird_call_and_bird_song"), None),
    "duck":          (L("Fowl", "Bird_vocalization_and_bird_call_and_bird_song"), L("duck", "quack")),
    "goose":         (L("Fowl", "Bird_vocalization_and_bird_call_and_bird_song"), L("goose", "honk")),
    "chicken":       (L("Chicken_and_rooster", "Fowl"), None),
    "parrot":        (L("Bird_vocalization_and_bird_call_and_bird_song"), L("parrot")),
    "rabbit":        (L("Domestic_animals_and_pets"), L("rabbit")),
    "hamster":       (L("Domestic_animals_and_pets"), L("hamster", "rodent")),
    "rat":           (L("Domestic_animals_and_pets"), L("rat", "rodent")),
    "frog":          (L("Frog"), None),
    "squirrel":      (L("Wild_animals"), L("squirrel")),
    "deer":          (L("Wild_animals"), L("deer")),
    "fox":           (L("Wild_animals"), L("fox")),
    "wolf":          (L("Wild_animals", "Growling"), L("wolf", "howl")),
    "monkey":        (L("Wild_animals"), L("monkey")),
    "pig":           (L("Livestock_and_farm_animals_and_working_animals"), L("pig", "oink")),
    "goat":          (L("Livestock_and_farm_animals_and_working_animals"), L("goat", "bleat")),
    "balloon":       (L("Crack"), L("balloon", "pop")),
    "guitar":        (L("Guitar", "Acoustic_guitar", "Electric_guitar", "Strum", "Plucked_string_instrument"), None),
    "piano":         (L("Piano"), None),
    "drum":          (L("Drum_kit", "Drum", "Snare_drum", "Bass_drum"), None),
    "soccer ball":   (L("Thump_and_thud"), L("soccer", "ball", "kick")),
    "basketball":    (L("Thump_and_thud", "Hands"), L("basketball", "dribble", "bounce")),
    "tennis ball":   (L("Thump_and_thud"), L("tennis", "bounce")),
    "folding chair": (L("Wood", "Squeak"), L("chair")),
    "step stool":    (L("Wood"), None),
}

# materials: loops + one-shot events
MATERIALS = {
    "WOOD":       (L("Wood", "Knock"),                        None, L("Knock")),
    "METAL":      (L("Hammer", "Cutlery_and_silverware", "Keys_jangling", "Ratchet_and_pawl", "Chink_and_clink"), None, L("Chink_and_clink", "Hammer")),
    "GLASS":      (L("Glass", "Chink_and_clink", "Shatter"),  None, L("Chink_and_clink", "Shatter")),
    "CERAMIC":    (L("Dishes_and_pots_and_pans", "Chink_and_clink"), None, L("Chink_and_clink")),
    "ENGINE":     (L("Engine", "Idling", "Car", "Motor_vehicle_(road)", "Accelerating_and_revving_and_vroom"), None, L("Engine_starting", "Vehicle_horn_and_car_horn_and_honking")),
    "WATER":      (L("Water", "Pour", "Water_tap_and_faucet", "Stream", "Drip", "Splash_and_splatter", "Waves_and_surf", "Rain", "Liquid"), None, L("Splash_and_splatter", "Drip")),
    "ELECTRONICS":(L("Buzz", "Computer_keyboard", "Typing", "Ringtone", "Alarm", "Tick"), None, L("Alarm", "Ringtone", "Buzz")),
    "FABRIC":     (L("Zipper_(clothing)", "Crumpling_and_crinkling", "Tearing"), None, L("Tearing", "Zipper_(clothing)")),
    "PLANT":      (L("Wind"), L("leaves", "foliage", "rustling", "grass", "forest"), L("Whoosh_and_swoosh_and_swish")),
    "PAPER":      (L("Crumpling_and_crinkling", "Tearing", "Writing", "Typewriter"), None, L("Tearing", "Crumpling_and_crinkling")),
    "PERSON":     (L("Speech", "Conversation", "Chatter", "Walk_and_footsteps", "Laughter", "Clapping", "Breathing"), None, L("Finger_snapping", "Clapping", "Knock")),
    "ANIMAL":     (L("Dog", "Cat", "Bark", "Meow", "Domestic_animals_and_pets", "Livestock_and_farm_animals_and_working_animals", "Wild_animals"), None, L("Bark", "Meow")),
    "BIRD":       (L("Bird", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Crow", "Gull_and_seagull", "Fowl", "Chicken_and_rooster"), None, L("Chirp_and_tweet", "Crow")),
    "FOOD":       (L("Chewing_and_mastication", "Frying_(food)", "Boiling", "Crushing"), None, L("Crushing", "Crack")),
    "NEUTRAL":    (L("Mechanisms", "Whoosh_and_swoosh_and_swish", "Rattle", "Wind", "Tick"), None, L("Whoosh_and_swoosh_and_swish", "Thump_and_thud", "Bell")),
}

# ---- relax over-strict tag filters / broaden labels --------------------
_RELAX_TAGS = {
    "tv", "television", "vacuum cleaner", "washing machine", "dishwasher",
    "blender", "food processor", "coffee maker", "computer monitor",
    "desktop computer", "box", "mirror", "lamp", "table lamp", "floor lamp",
    "ceiling light", "hair drier", "tray", "strainer", "grater", "whisk",
    "rolling pin", "ambulance", "police car", "fire truck", "taxi", "van",
    "tractor", "excavator", "crane", "forklift", "kick scooter", "scooter",
    "tree", "bush", "flower", "elevator", "escalator", "stapler", "magazine",
    "envelope", "pill bottle", "napkin", "paper towel roll",
    # food & drinks: share the big Chewing/Pour pools, tags kill them
    "banana", "apple", "sandwich", "orange", "carrot", "pizza", "hot dog",
    "donut", "cake", "bread", "pastry", "croissant", "cookie", "chocolate",
    "candy", "cheese", "milk", "juice", "honey", "jam", "ketchup", "sauce",
    "soup", "pasta", "rice", "cereal", "sausage", "ham", "meat", "fish",
    "shrimp", "sushi", "burger", "french fries", "pancakes", "pie",
    "ice cream", "watermelon", "grape", "strawberry", "tomato", "cucumber",
    "potato", "onion", "garlic", "bell pepper", "lemon", "peach", "pear",
    "pineapple", "mango", "cabbage", "lettuce", "corn", "mushroom",
    "avocado", "pumpkin", "eggplant", "nuts", "salad",
}
_EXTRA_LABELS = {
    "tv": ["Domestic_sounds_and_home_sounds"],
    "television": ["Domestic_sounds_and_home_sounds"],
    "vacuum cleaner": ["Domestic_sounds_and_home_sounds", "Mechanisms"],
    "washing machine": ["Domestic_sounds_and_home_sounds", "Mechanisms"],
    "dishwasher": ["Domestic_sounds_and_home_sounds", "Mechanisms"],
    "blender": ["Domestic_sounds_and_home_sounds"],
    "food processor": ["Domestic_sounds_and_home_sounds"],
    "coffee maker": ["Domestic_sounds_and_home_sounds"],
    "computer monitor": ["Domestic_sounds_and_home_sounds"],
    "desktop computer": ["Domestic_sounds_and_home_sounds"],
    "box": ["Domestic_sounds_and_home_sounds"],
    "elephant": ["Animal"],
    "bear": ["Animal"],
    "zebra": ["Animal"],
    "giraffe": ["Animal"],
    "monkey": ["Animal"],
    "squirrel": ["Animal"],
    "deer": ["Animal"],
    "fox": ["Animal"],
    "rabbit": ["Animal"],
    "hamster": ["Animal"],
    "rat": ["Animal"],
}
for _t in _RELAX_TAGS:
    if _t in MAPPING:
        _lb, _tg = MAPPING[_t]
        MAPPING[_t] = (_lb, None)
for _t, _ex in _EXTRA_LABELS.items():
    if _t in MAPPING:
        _lb, _tg = MAPPING[_t]
        MAPPING[_t] = (_lb + _ex, _tg)

def norm_lic(url: str) -> str:
    u = (url or "").lower()
    if "publicdomain" in u or "zero" in u:
        return "CC0"
    if "by-nc" in u:
        return "CC-BY-NC"
    if "/by/" in u:
        return "CC-BY"
    if "sampling" in u:
        return "CC-SAMPLING"
    return "OTHER"

def main():
    lt = json.load(open(os.path.join(DATA, "labeltable.json")))
    EN, MAT = lt["EN"], lt["MAT"]
    name_to_idx = defaultdict(list)
    for i, n in enumerate(EN):
        name_to_idx[n].append(i)

    rows = []
    for fn in ("eval.csv", "dev.csv"):
        with open(os.path.join(DATA, fn), newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            fi = header.index("fname"); li = header.index("labels")
            for row in r:
                if not row or not row[fi]:
                    continue
                rows.append((row[fi], frozenset(row[li].split(","))))
    print("clips with labels:", len(rows))

    info = {}
    for fn in ("eval_clips_info.json", "dev_clips_info.json"):
        info.update(json.load(open(os.path.join(DATA, fn))))
    print("clip info:", len(info))

    pnp = json.load(open(os.path.join(DATA, "pp_pnp.json")))

    by_label = defaultdict(list)
    for cid, labels in rows:
        for lb in labels:
            by_label[lb].append(cid)

    vocab_names = set()
    with open(os.path.join(DATA, "vocabulary.csv"), newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                vocab_names.add(row[1])

    candidates = {}
    used = {}
    stats = []

    def collect(target, labels, tags_any, mat=None, kind="class"):
        pool = []
        seen = set()
        for lb in labels:
            if lb not in vocab_names:
                print(f"  !! label not in vocab: {lb} (target {target})")
                continue
            for cid in by_label.get(lb, ()):
                if cid in seen:
                    continue
                seen.add(cid)
                meta = info.get(cid)
                if not meta:
                    continue
                lic = norm_lic(meta.get("license", ""))
                if lic not in ("CC0", "CC-BY"):
                    continue
                text = " ".join([meta.get("title", ""), meta.get("description", ""),
                                 " ".join(meta.get("tags", []))]).lower()
                if tags_any and not any(t in text for t in tags_any):
                    continue
                promo = 0.5
                ent = pnp.get(cid)
                if ent:
                    vals = []
                    for v in ent.values():
                        vals.extend([x for x in v if x >= 0])
                    if vals:
                        promo = max(promo, max(vals))
                if tags_any:
                    promo += 0.25 * sum(1 for t in tags_any if t in text)
                pool.append({"id": cid, "lic": lic, "promo": promo,
                             "uploader": meta.get("uploader", "?"),
                             "title": meta.get("title", "")[:80]})
        pool.sort(key=lambda c: (0 if c["lic"] == "CC0" else 1, -c["promo"]))
        picked, c0, cb = [], 0, 0
        for c in pool:
            if c["id"] in used:
                continue
            if c["lic"] == "CC0" and c0 >= MAX_CC0:
                continue
            if c["lic"] == "CC-BY" and cb >= MAX_CCBY:
                continue
            picked.append(c)
            used[c["id"]] = target
            if c["lic"] == "CC0":
                c0 += 1
            else:
                cb += 1
            if len(picked) >= MAX_PER_TARGET:
                break
        candidates[target] = {"kind": kind, "material": mat, "clips": picked}
        stats.append((target, len(pool), len(picked), c0, cb))

    for target, (labels, tags) in MAPPING.items():
        idxs = name_to_idx.get(target, [])
        mat = MAT[idxs[0]] if idxs else "NEUTRAL"
        collect(target, labels, tags, mat=mat, kind="class")
    for mat, (labels, tags, _ev) in MATERIALS.items():
        collect(mat, labels, tags, mat=mat, kind="material")

    json.dump(candidates, open(OUT, "w"), ensure_ascii=False)
    print(f"\ntargets: {len(candidates)}  unique clips to download: {len(used)}")
    poor = [(t, n) for t, _, n, _, _ in stats if n < 8]
    print("targets with <8 candidates:", poor)
    with open(os.path.join(DATA, "candidates_stats.txt"), "w") as f:
        for t, npool, npick, c0, cb in sorted(stats):
            f.write(f"{t:20s} pool={npool:5d} picked={npick:3d} cc0={c0:3d} ccby={cb:3d}\n")

if __name__ == "__main__":
    main()
