# Earthquake Forecasting #

### What is this repository for? ###

this project designed to create an early-warning earthquake prediction system using a supervised neural network, orignally the task goal & documentation was from a topcoder marathon match challenge [here](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16510&pm=13913)

## training set ##
* 75 test scenarios from multiple countries, notably Peru, Indonesia, and southern California.

* each test case contains 2160 hour time steps (90 days), which each contain sensor data on a per second, per sample rate basis (number of sensor seps per hr is 3600*sampleRate)
- sensor data consists of 3 channels, each channel contains a different analog sensor reading.

* the number of sites for each test scenario varies between 5-9, and the lat/lon of each site is given.

* besides the 3 channel site data, each test also gives an "other global quakes" occuring during that hour, with their magnitudes, lat/lon, and fracture depth given, as well the planetary electromagnetic activity index (denoted Kp in the code) is given on a per hour basis.

## answer key ##
* The hour that the quake occurs, which is 0<x<=2160, where X is the hour of the quake event.
* the magnitude of the quake is given, 0<x<=10, X is the magnitude of the quake event.
* latitude/longitude and depth of the quake event's hypocenter.
* distance to the nearest site is also given (km).

## scoring: ##
* the external scoring tool is a java jar applet that runs the project itself (given to me by the topCoder challenge admins), scoring begins after the 768th hour and follow the below forumula:


```
#!c++

S = sizeof(NN) * (2 * G * - sum of squared values in NN) -1

```
### where: ###
* S is the score for a particular test case.
* NN is the normalized submatrix of your returned matrix N, within the hours of 768> x <=2160.
* G is the hour of the quake event.

* the internal scoring method follows a different approach, as the original scoring was quite rudamentary compared to the scope of the project (it doesn't say where the quake is, it only cares about when, which is naieve as the inter-site distances could be quite large), following the below formula

```
#!c++

S = (oldFit*2 + exp(-(fabs(whenGuess-whenAns)+distCalc(latGuess, lonGuess, latAns, lonAns))))/3
```
### where: ###
 * S is the score for a particular time step (hour)
 * oldFit is the previous iterations fitness for this particular individual.
 * whenGuess is the guessed hour for this timestep, for this particular individual.
 * whenAns is the hour of the quake event provided by the answer key.
 * lat/lon Guess is the predicted nearest site to the quake event, for this particular individual.
 * lat/long ans is the nearest site to the quake, provided by the answer key.
 * fabs is a double precison absolute value function.
 * distCalc calculates the scalar distance between coordinates on an oblate spheroid (the earth)

### How do I get set up? ###

* Summary of set up
* Configuration
requires cmake 3.2.1 or newer to configure
* Dependencies
requires cuda toolkit 7.0 or newer to compile
requires a cuda device architecture of 3.0 or newer to run
* Deployment instructions
run the java scoring tool located [here](http://www.topcoder.com/contest/problem/QuakePredictor/manual.html), rather than the compiled program directly.