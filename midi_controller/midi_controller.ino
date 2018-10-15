
//libraries
#include <JC_Button.h> // https://github.com/JChristensen/JC_Button
#include <MIDI.h>
MIDI_CREATE_DEFAULT_INSTANCE();

//constants
static const unsigned ledPin = 13;      // LED pin on Arduino Uno
const int potPin = A0; //pin A0 to read analog input
const byte BUTTON_PIN(7); //pin connected to button switch
const int vel = 100; //constant velocity for notes TEMPORARILY ADD POTI

//variables
int potiValue; //save analog value
int current_note;

//classes
Button btn(BUTTON_PIN, 25, true, false);       // define the button

void setup()
{
    //pinMode(ledPin, OUTPUT);
    btn.begin();                       // initialize button
    MIDI.begin(4);                    // Launch MIDI and listen to channel 4
    //Serial.begin(9600);
}

void loop()
{
    btn.read();                        //read button status
    potiValue = analogRead(potPin);             //read and save analog value from potentiometer 
    potiValue = map(potiValue, 0, 1023, 95, 36);    //map value 0-1023 to 36-95 (PWM)  
    //Serial.println(potiValue);
    if(btn.wasPressed()){                   //read poti and send note on msg
      MIDI.sendNoteOn(potiValue, vel, 1);         //send note on with poti value on channel 1
      //Serial.print("Note on: ");
      //Serial.println(potiValue);
      current_note = potiValue;
    }

    if(btn.wasReleased()){                            //take saved poti value and send note off msg
      MIDI.sendNoteOff(current_note, 0, 1);           //send note off msg
      //Serial.print("Note off: ");
      //Serial.println(potiValue);
    }

    if(potiValue != current_note && btn.isPressed()){
      MIDI.sendNoteOff(current_note, 0, 1);         //send note off msg
      MIDI.sendNoteOn(potiValue, vel, 1);
      current_note = potiValue;
      delay(5);
    }
}

