#include "Arduino_BMI270_BMM150.h"

// Define motor control pins
const int in1A = 2;  // Motor A direction pin 1
const int in2A = 3;  // Motor A direction pin 2
const int in1B = 4;  // Motor B direction pin 1
const int in2B = 5;  // Motor B direction pin 2
const int pwmA = 9;  // Motor A PWM pin
const int pwmB = 10; // Motor B PWM pin

float ang_degrees = 0;

// Angular PID constants
const float ang_Kp = 2.0;
const float ang_Ki = 0.1;
const float ang_Kd = 0.5;

// Angular PID variables
float ang_previous_error = 0;
float ang_integral = 0;

void setup() 
{
  Serial.begin(9600);
  imu_setup();
  motor_setup();
}

void loop()
{
  // Example function calls
  turn(90);  // Turn left 90 degrees
  delay(2000);  // Delay between movements for testing

  turn(-90);  // Turn right 90 degrees
  delay(2000);

  turn(180);  // Turn back (left)
  delay(2000);

  turn(-180);  // Turn back (right)
  delay(2000);
}

void imu_setup()
{
  if (!IMU.begin()) 
  {
    Serial.println("Failed to initialize IMU!");
    while (1); // Halt the program if IMU fails to initialize
  }
}

float angular_displacement()
{
  float x, y, z;
  float sampling_time = 1.0 / IMU.gyroscopeSampleRate(); 

  if (IMU.gyroscopeAvailable()) 
  {
    IMU.readGyroscope(x, y, z); 

    // To remove zero error
    if (abs(z) > 1) 
    {
      ang_degrees += sampling_time * z; // Update degrees with filtered z
    }
  }
  return ang_degrees;
}

void turn(float target_angle)
{
  float current_angle = angular_displacement();
  float angular_target = current_angle + target_angle;
  float ang_error;
  float last_time = millis();
  float dt;

  ang_integral = 0;  // Reset integral at the start of each turn
  ang_previous_error = 0;

  do
  {
    float current_time = millis();
    dt = (current_time - last_time) / 1000.0;  // Convert to seconds
    last_time = current_time;

    ang_error = angular_target - angular_displacement();
    
    // PID calculation
    ang_integral += ang_error * dt;
    float ang_derivative = (ang_error - ang_previous_error) / dt;
    float output = ang_Kp * ang_error + ang_Ki * ang_integral + ang_Kd * ang_derivative;

    // Limit the output to a reasonable range (e.g., -255 to 255)
    output = constrain(output, -255, 255);

    // Determine direction based on the sign of the output
    int direction = (output > 0) ? 1 : -1;

    // Use the absolute value of output for speed
    int speed = abs(output);

    // Control motors
    control_motors(-direction, direction, speed, speed);

    ang_previous_error = ang_error;
    delay(10); // Short delay to avoid rapid changes
  } while (abs(ang_error) >= 2);  // Tighter tolerance for more accurate turns

  // Stop motors after completing the turn
  control_motors(0, 0, 0, 0);
}

void motor_setup() 
{
  pinMode(in1A, OUTPUT);
  pinMode(in2A, OUTPUT);
  pinMode(in1B, OUTPUT);
  pinMode(in2B, OUTPUT);
  pinMode(pwmA, OUTPUT);
  pinMode(pwmB, OUTPUT);

  // Set initial states
  digitalWrite(in1A, LOW);
  digitalWrite(in2A, LOW);
  digitalWrite(in1B, LOW);
  digitalWrite(in2B, LOW);
  analogWrite(pwmA, 0);
  analogWrite(pwmB, 0);
}

void control_motors(int directionA, int directionB, int speedA, int speedB) 
{
  // Limit the speeds to a maximum of 255
  speedA = constrain(speedA, 0, 255);
  speedB = constrain(speedB, 0, 255);

  // Control Motor A direction and speed
  digitalWrite(in1A, directionA > 0 ? HIGH : LOW);
  digitalWrite(in2A, directionA < 0 ? HIGH : LOW);
  
  // Control Motor B direction and speed
  digitalWrite(in1B, directionB > 0 ? HIGH : LOW);
  digitalWrite(in2B, directionB < 0 ? HIGH : LOW);

  // Set the motor speeds
  analogWrite(pwmA, speedA);
  analogWrite(pwmB, speedB);
}
