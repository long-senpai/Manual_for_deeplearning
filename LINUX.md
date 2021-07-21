# HOW TO SWITCH BETWEEN MULTIPLE GCC, G++ VERSION

### Use the update-alternatives tool to create list of multiple GCC and G++ compiler alternatives: 
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```
### Check the available C and C++ compilers list on your Ubuntu 20.04 system and select desired version by entering relevant selection number: 
```
$ sudo update-alternatives --config gcc
There are 3 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path            Priority   Status
------------------------------------------------------------
  0            /usr/bin/gcc-9   9         auto mode
  1            /usr/bin/gcc-7   7         manual mode
* 2            /usr/bin/gcc-8   8         manual mode
  3            /usr/bin/gcc-9   9         manual mode
Press  to keep the current choice[*], or type selection number: 



$ sudo update-alternatives --config g++
There are 3 choices for the alternative g++ (providing /usr/bin/g++).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/g++-9   9         auto mode
  1            /usr/bin/g++-7   7         manual mode
  2            /usr/bin/g++-8   8         manual mode
  3            /usr/bin/g++-9   9         manual mode

Press  to keep the current choice[*], or type selection number:
``` 
### LINUX COMMAND:
- **ldconfig**: 
    ldconfig is a program that is used to maintain the shared library cache. This cache is typically stored in the file /etc/ld.so.cache and is used by the system to map a shared library name to the location of the corresponding shared library file
   

