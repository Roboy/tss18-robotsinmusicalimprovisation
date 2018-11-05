// Arduino Button Library
// https://github.com/JChristensen/JC_Button
// Copyright (C) 2018 by Jack Christensen and licensed under
// GNU GPL v3.0, https://www.gnu.org/licenses/gpl.html
//
// Example sketch to turn an LED on and off with a tactile button switch.
// Wire the switch from the Arduino pin to ground.

#include <JC_Button.h>          // https://github.com/JChristensen/JC_Button

// pin assignments
const byte
    BUTTON_RED(2),
    BUTTON1(7),              // connect a button switch from this pin to ground
    BUTTON2(4),
    BUTTON3(5);

Button redBtn(BUTTON_RED, 25, true, false);       // define the button
Button btn1(BUTTON1, 25, true, false);
Button btn2(BUTTON2, 25, true, false);
Button btn3(BUTTON3, 25, true, false);

void setup()
{
    redBtn.begin();
    btn1.begin();
    btn2.begin();
    btn3.begin();
    Serial.begin(9600);
}

void loop()
{
    redBtn.read();
    btn1.read();
    btn2.read();
    btn3.read();

    if (redBtn.wasPressed())
    {
      Serial.println("Red Button was pressed");
    }
    if (btn1.wasPressed())    // if the button was released, change the LED state
    {
      Serial.println("Button 1 was pressed");
    }
    if (btn2.wasPressed())    // if the button was released, change the LED state
    {
      Serial.println("Button 2 was pressed");
    }
    if(btn3.wasPressed()){
      Serial.println("Button 3 was pressed");
    }
}

