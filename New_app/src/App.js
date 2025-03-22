import React, { useEffect, useRef } from "react";
import { View, Text, Button } from "react-native";
import KeyEvent from "react-native-keyevent"; // Detect hardware button press
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import DetectScreen from "./DetectScreen"; // Import DetectScreen

const Stack = createStackNavigator();

const App = () => {
    const pressTimer = useRef(null);

    useEffect(() => {
        KeyEvent.onKeyDownListener((keyEvent) => {
            if (keyEvent.keyCode === 24) { // Volume Up key (keycode 24)
                pressTimer.current = setTimeout(() => {
                    navigationRef.current?.navigate("DetectScreen");
                }, 1000); // Trigger after 1 second
            }
        });

        KeyEvent.onKeyUpListener(() => {
            clearTimeout(pressTimer.current);
        });

        return () => {
            KeyEvent.removeKeyDownListener();
            KeyEvent.removeKeyUpListener();
        };
    }, []);

    return (
        <NavigationContainer ref={navigationRef}>
            <Stack.Navigator>
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="DetectScreen" component={DetectScreen} />
            </Stack.Navigator>
        </NavigationContainer>
    );
};

const navigationRef = React.createRef();

const HomeScreen = ({ navigation }) => {
    return (
        <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
            <Text>Press Volume Up to Open DetectScreen</Text>
            <Button title="Go to DetectScreen" onPress={() => navigation.navigate("DetectScreen")} />
        </View>
    );
};

export default App;
