
//POTI TEST
//Constants:
const int potPin = A0; //pin A0 to read analog input


//Variables:
int value; //save analog value

void setup(){
  Serial.begin(9600);
}

void loop(){
  //Read and save analog value from potentiometer
  value = analogRead(potPin);   
  //Map value 0-1023 to 36-95 (PWM)       
  value = map(value, 0, 1023, 95, 36); 
  Serial.println(value);
  delay(100);                          //Small delay
}



