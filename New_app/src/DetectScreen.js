import React, { useState } from "react";
import { View, Text, TouchableOpacity, Image, ActivityIndicator, Alert } from "react-native";
import ImagePicker from "react-native-image-picker";
import axios from "axios";
import Tts from "react-native-tts";  // Add TTS

const API_URL = "http://your-server-ip:5000/detect"; // Change to your Flask API URL

const DetectScreen = () => {
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const pickImage = () => {
        const options = { mediaType: "photo", quality: 1 };
        ImagePicker.showImagePicker(options, (response) => {
            if (!response.didCancel && !response.error) {
                setImage(response.uri);
                uploadImage(response);
            }
        });
    };

    const uploadImage = async (file) => {
        setLoading(true);
        const formData = new FormData();
        formData.append("file", { uri: file.uri, type: file.type || "image/jpeg", name: "image.jpg" });

        try {
            const response = await axios.post(API_URL, formData, {
                headers: { "Content-Type": "multipart/form-data" }, // No API key needed
            });

            setResult(response.data.detections);
            speakResults(response.data.detections);
        } catch (error) {
            Alert.alert("Error", "Failed to process image.");
            console.error("Upload Error:", error);
        } finally {
            setLoading(false);
        }
    };

    const speakResults = (detections) => {
        if (detections.length > 0) {
            let speechText = detections.map((item) => item.sentence).join(", ");
            Tts.speak(speechText);
        } else {
            Tts.speak("No objects detected.");
        }
    };

    return (
        <View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: 20 }}>
            <TouchableOpacity onPress={pickImage} style={{ backgroundColor: "#007bff", padding: 15, borderRadius: 10 }}>
                <Text style={{ color: "#fff", fontSize: 18 }}>Select Image</Text>
            </TouchableOpacity>

            {image && <Image source={{ uri: image }} style={{ width: 200, height: 200, margin: 20 }} />}
            {loading && <ActivityIndicator size="large" color="#007bff" />}
            {result && (
                <View>
                    <Text style={{ fontSize: 18, fontWeight: "bold", marginBottom: 10 }}>Detection Results:</Text>
                    {result.length > 0 ? result.map((item, index) => (
                        <Text key={index} style={{ fontSize: 16 }}>{item.sentence}</Text>
                    )) : <Text>No objects detected.</Text>}
                </View>
            )}
        </View>
    );
};

export default DetectScreen;
