#include <HardwareSerial.h>

//sparkfun midi chip definitions
#define PIN_RAW_INPUT 4

//Potis not in use at the moment
#define PIN_POT_A0 0
#define PIN_POT_A1 1

#define MIDI_BAUD_RATE 31250

void setup()
{
  Serial.begin(MIDI_BAUD_RATE);
  Serial1.begin(MIDI_BAUD_RATE);

  pinMode(PIN_RAW_INPUT, INPUT_PULLUP);
}

void loop()
{ 

  if(digitalRead(PIN_RAW_INPUT) == LOW)
  {
    byte input;
    if(Serial1.available() != 0)
    {   
      input = Serial1.read();
      Serial.write(input);
    }
  } 
}
