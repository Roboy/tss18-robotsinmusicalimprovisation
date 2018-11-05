
//libraries
#include <JC_Button.h> // https://github.com/JChristensen/JC_Button
#include <MIDI.h>
MIDI_CREATE_DEFAULT_INSTANCE();

//constants
const int 
    faderPin1 = A5,
    faderPin2 = A4,
    faderPin3 = A3,
    potPin1 = A1,
    potPin2 = A0,
    potPin3 = A2;
    
const byte
    BUTTON_EOS(2),
    BUTTON1(7),
    BUTTON2(4),
    BUTTON3(5);
    
const int vel = 100; //constant velocity for notes TEMPORARILY ADD POTI

//variables
int 
    fader1,
    fader2,
    fader3,
    poti1,
    poti2,
    poti3,
    currentNote1,
    currentNote2,
    currentNote3;

//classes
Button eosBtn(BUTTON_EOS, 25, true, false);       // define the button
Button btn1(BUTTON1, 25, true, false);
Button btn2(BUTTON2, 25, true, false);
Button btn3(BUTTON3, 25, true, false);

void setup()
{
    MIDI.begin(4);                    // Launch MIDI and listen to channel 4
    eosBtn.begin();
    btn1.begin();
    btn2.begin();
    btn3.begin();
    //Serial.begin(9600);
}

void loop()
{
    //read and save analog value from potis
    //and then map value 0-1023 to 36-95 (PWM)  
    fader1 = analogRead(faderPin1);             
    fader1 = map(fader1, 0, 1023, 36, 95);    
    fader2 = analogRead(faderPin2);
    fader2 = map(fader2, 0, 1023, 36, 95);
    fader3 = analogRead(faderPin3);
    fader3 = map(fader3, 0, 1023, 36, 95);
    poti1 = analogRead(potPin1);
    poti1 = map(poti1, 0, 1023, 127, 0);
    poti2 = analogRead(potPin2);
    poti2 = map(poti2, 0, 1023, 127, 0);
    poti3 = analogRead(potPin3);
    poti3 = map(poti3, 0, 1023, 127, 0);
 
    //end of sequence button
    eosBtn.read();
    if (eosBtn.wasPressed()){
      MIDI.sendNoteOn(127, vel, 1);
    }
    if (eosBtn.wasReleased()){
      MIDI.sendNoteOff(127, 0, 1);
    }
 
    //button 1
    btn1.read();
    if (btn1.wasPressed()){
      MIDI.sendNoteOn(fader1, poti1, 1);
      currentNote1 = fader1;
    }
    if (btn1.wasReleased()){
      MIDI.sendNoteOff(currentNote1, 0, 1);
    }
    if(fader1 != currentNote1 && btn1.isPressed()){
      MIDI.sendNoteOff(currentNote1, 0, 1);         //send note off msg
      MIDI.sendNoteOn(fader1, poti1, 1);
      currentNote1 = fader1;
      delay(5);
    }
 
    //button 2 == major chord
    btn2.read();
    if(btn2.wasPressed()){
      MIDI.sendNoteOn(fader2, poti2, 1);
      MIDI.sendNoteOn(fader2+4, poti2, 1);
      MIDI.sendNoteOn(fader2+7, poti2, 1);
      currentNote2 = fader2;
    }                       
    if(btn2.wasReleased()){
      MIDI.sendNoteOff(currentNote2, 0, 1);
      MIDI.sendNoteOff(currentNote2+4, 0, 1);
      MIDI.sendNoteOff(currentNote2+7, 0, 1);
    } 
      
    //button 3 == minor chord
    btn3.read();
    if(btn3.wasPressed()){
      MIDI.sendNoteOn(fader3, poti3, 1);
      MIDI.sendNoteOn(fader3+3, poti3, 1);
      MIDI.sendNoteOn(fader3+7, poti3, 1);
      currentNote3 = fader3;
    }                       
    if(btn3.wasReleased()){
      MIDI.sendNoteOff(currentNote3, 0, 1);
      MIDI.sendNoteOff(currentNote3+3, 0, 1);
      MIDI.sendNoteOff(currentNote3+7, 0, 1);
    } 
}

