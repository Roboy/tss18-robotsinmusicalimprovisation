#include <Arduino.h>
#include <SPI.h>
#include <JC_Button.h> // https://github.com/JChristensen/JC_Button
#include "Adafruit_BLE.h"
#include "Adafruit_BluefruitLE_SPI.h"
#include "Adafruit_BluefruitLE_UART.h"
#include "Adafruit_BLEMIDI.h"
#if SOFTWARE_SERIAL_AVAILABLE
  #include <SoftwareSerial.h>
#endif

#include "BluefruitConfig.h"

#define FACTORYRESET_ENABLE         0
#define MINIMUM_FIRMWARE_VERSION    "0.7.0"

// This app was tested on iOS with the following apps:
//
// https://itunes.apple.com/us/app/midimittr/id925495245?mt=8
// https://itunes.apple.com/us/app/igrand-piano-free-for-ipad/id562914032?mt=8
//
// To test:
// - Run this sketch and open the Serial Monitor
// - Open the iGrand Piano Free app
// - Open the midimittr app on your phone and under Clients select "Adafruit Bluefruit LE"
// - When you see the 'Connected' label switch to the Routing panel
// - Set the Destination to 'iGrand Piano'
// - Switch to the iGrand Piano Free app and you should see notes playing one by one

// Create the bluefruit object, either software serial...uncomment these lines
/*
SoftwareSerial bluefruitSS = SoftwareSerial(BLUEFRUIT_SWUART_TXD_PIN, BLUEFRUIT_SWUART_RXD_PIN);

Adafruit_BluefruitLE_UART ble(bluefruitSS, BLUEFRUIT_UART_MODE_PIN,
                              BLUEFRUIT_UART_CTS_PIN, BLUEFRUIT_UART_RTS_PIN);
*/

/* ...or hardware serial, which does not need the RTS/CTS pins. Uncomment this line */
// Adafruit_BluefruitLE_UART ble(BLUEFRUIT_HWSERIAL_NAME, BLUEFRUIT_UART_MODE_PIN);

/* ...hardware SPI, using SCK/MOSI/MISO hardware SPI pins and then user selected CS/IRQ/RST */
 Adafruit_BluefruitLE_SPI ble(BLUEFRUIT_SPI_CS, BLUEFRUIT_SPI_IRQ, BLUEFRUIT_SPI_RST);

/* ...software SPI, using SCK/MOSI/MISO user-defined SPI pins and then user selected CS/IRQ/RST */
//Adafruit_BluefruitLE_SPI ble(BLUEFRUIT_SPI_SCK, BLUEFRUIT_SPI_MISO,
//                             BLUEFRUIT_SPI_MOSI, BLUEFRUIT_SPI_CS,
//                             BLUEFRUIT_SPI_IRQ, BLUEFRUIT_SPI_RST);

Adafruit_BLEMIDI midi(ble);

bool isConnected = false;
int current_note = 60;

const byte 
  BUTTON1(13),
  BUTTON2(12),
  BUTTON3(11),
  BUTTON_EOS(10);

const int 
    faderPin1 = A2,
    faderPin2 = A1,
    faderPin3 = A0,
    potPin1 = A3,
    potPin2 = A4,
    potPin3 = A5,
    bleLED = 9;

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
    
Button eosBtn(BUTTON_EOS, 25, true, false);
Button btn1(BUTTON1, 25, true, false);
Button btn2(BUTTON2, 25, true, false);
Button btn3(BUTTON3, 25, true, false);


// A small helper
void error(const __FlashStringHelper*err) {
  Serial.println(err);
  while (1);
}

// callback
void connected(void)
{
  isConnected = true;
  
  Serial.println(F(" CONNECTED!"));
  digitalWrite(bleLED, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);

}

void disconnected(void)
{
  Serial.println("disconnected");
  digitalWrite(bleLED, LOW);
  isConnected = false;
}

void BleMidiRX(uint16_t timestamp, uint8_t status, uint8_t byte1, uint8_t byte2)
{
  Serial.print("[MIDI ");
  Serial.print(timestamp);
  Serial.print(" ] ");

  Serial.print(status, HEX); Serial.print(" ");
  Serial.print(byte1 , HEX); Serial.print(" ");
  Serial.print(byte2 , HEX); Serial.print(" ");

  Serial.println();
}

void setup(void)
{
  eosBtn.begin();
  btn1.begin();
  btn2.begin();
  btn3.begin();
  pinMode(bleLED, OUTPUT);

  // while (!Serial);  // required for Flora & Micro
  delay(500);
  
  Serial.begin(115200);
  Serial.println(F("Adafruit Bluefruit MIDI Example"));
  Serial.println(F("---------------------------------------"));

  /* Initialise the module */
  Serial.print(F("Initialising the Bluefruit LE module: "));

  if ( !ble.begin(VERBOSE_MODE) )
  {
    error(F("Couldn't find Bluefruit, make sure it's in CoMmanD mode & check wiring?"));
  }
  Serial.println( F("OK!") );

  if ( FACTORYRESET_ENABLE )
  {
    /* Perform a factory reset to make sure everything is in a known state */
    Serial.println(F("Performing a factory reset: "));
    if ( ! ble.factoryReset() ) {
      error(F("Couldn't factory reset"));
    }
  }

  //ble.sendCommandCheckOK(F("AT+uartflow=off"));
  ble.echo(false);

  Serial.println("Requesting Bluefruit info:");
  /* Print Bluefruit information */
  ble.info();

  /* Set BLE callbacks */
  ble.setConnectCallback(connected);
  ble.setDisconnectCallback(disconnected);

  // Set MIDI RX callback
  //midi.setRxCallback(BleMidiRX);

  Serial.println(F("Enable MIDI: "));
  if ( ! midi.begin(true) )
  {
    error(F("Could not enable MIDI"));
  }

  ble.verbose(false);
  Serial.print(F("Waiting for a connection..."));
}

void loop(void)
{
  // interval for each scanning ~ 500ms (non blocking)
  ble.update(250);

  // bail if not connected
  if (! isConnected)
    return;

  fader1 = analogRead(faderPin1);             
  fader1 = map(fader1, 0, 1023, 95, 36);    
  fader2 = analogRead(faderPin2);
  fader2 = map(fader2, 0, 1023, 95, 36);
  fader3 = analogRead(faderPin3);
  fader3 = map(fader3, 0, 1023, 95, 36);
  poti1 = analogRead(potPin1);
  poti1 = map(poti1, 0, 1023, 0, 127);
  poti2 = analogRead(potPin2);
  poti2 = map(poti2, 0, 1023, 0, 127);
  poti3 = analogRead(potPin3);
  poti3 = map(poti3, 0, 1023, 0, 127);
  

   //eos button
  eosBtn.read();
  if (eosBtn.wasPressed()){
    // send note on
    Serial.print("Sending pitch ");
    Serial.println(current_note);
    midi.send(0x90, 127, 0x64);
  }
  if (eosBtn.wasReleased()){
    // send note off
    midi.send(0x80, 127, 0x64);
  }
  
  //button 1
  btn1.read();
  if (btn1.wasPressed()){
    // send note on
//    Serial.print("Sending pitch ");
//    Serial.println(current_note);
    midi.send(0x90, fader1, poti1);
    currentNote1 = fader1;
  }
  if (btn1.wasReleased()){
    // send note off
    midi.send(0x80, currentNote1, 0);
  }
  
    //button 2 == major chord
  btn2.read();
  if (btn2.wasPressed()){
    // send note on
//    Serial.print("Sending pitch ");
//    Serial.println(current_note);
    midi.send(0x90, fader2, poti2);
    midi.send(0x90, fader2+4, poti2);
    midi.send(0x90, fader2+7, poti2);
    midi.send(0x90, fader2+12, poti2);
    currentNote2 = fader2;
  }
  if (btn2.wasReleased()){
    // send note off
    midi.send(0x80, currentNote2, 0);
    midi.send(0x80, currentNote2 + 4, 0);
    midi.send(0x80, currentNote2 + 7, 0);
    midi.send(0x80, currentNote2 + 12, 0);
  }

    //button 3
  btn3.read();
  if (btn3.wasPressed()){
    // send note on
//    Serial.print("Sending pitch ");
//    Serial.println(current_note);
    midi.send(0x90, fader3, poti3);
    midi.send(0x90, fader3 + 3, poti3);
    midi.send(0x90, fader3 + 7, poti3);
    midi.send(0x90, fader3 + 12, poti3);
    currentNote3 = fader3;
  }
  if (btn3.wasReleased()){
    // send note off
    midi.send(0x80, currentNote3, 0);
    midi.send(0x80, currentNote3 + 3, 0);
    midi.send(0x80, currentNote3 + 7, 0);
    midi.send(0x80, currentNote3 + 12, 0);
  }


}
