#generate ping command from host to each ip address in the file ip.txt and ping 100 times per ip address

#!/bin/bash
while read line
do
ping -c 10 $line
done < ip.txt

