plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val ciKeystore = rootProject.file("signing/ci-release.jks")
val hasCiSigning = ciKeystore.exists()

android {
    namespace = "com.echolens.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.echolens.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 6
        versionName = "1.4.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    if (hasCiSigning) {
        signingConfigs {
            create("release") {
                storeFile = ciKeystore
                storePassword = "echolens-ci"
                keyAlias = "echolens"
                keyPassword = "echolens-ci"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (hasCiSigning) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += "tflite"
        // OGG samples must stay uncompressed so MediaExtractor can open them
        // directly through AssetFileDescriptor (sound texture library).
        noCompress += "ogg"
    }

    lint {
        abortOnError = false
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX — camera stream for object detection
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")

    // Foreground service with lifecycle
    implementation("androidx.lifecycle:lifecycle-service:2.8.4")

    // ONNX Runtime — YOLO26-N detector + Depth Anything V2 + optional RAVE backends
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // LiteRT (TensorFlow Lite) — Stable Audio Open Small on-device texture generation
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}
