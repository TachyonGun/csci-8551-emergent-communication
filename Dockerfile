FROM pytorch/pytorch:latest

RUN apt-get update

WORKDIR /home
ADD ./docker/requirements.txt .

RUN pip install -r requirements.txt

#_libgcc_mutex             0.1                        main
#_openmp_mutex             4.5                       1_gnu
#ca-certificates           2021.10.26           h06a4308_2
#certifi                   2021.10.8        py38h06a4308_0
#cloudpickle               2.0.0                    pypi_0    pypi
#cycler                    0.11.0                   pypi_0    pypi
#gym                       0.21.0                   pypi_0    pypi
#kiwisolver                1.3.2                    pypi_0    pypi
#ld_impl_linux-64          2.35.1               h7274673_9
#libffi                    3.3                  he6710b0_2
#libgcc-ng                 9.3.0               h5101ec6_17
#libgomp                   9.3.0               h5101ec6_17
#libstdcxx-ng              9.3.0               hd4cf53a_17
#matplotlib                3.4.3                    pypi_0    pypi
#ncurses                   6.3                  h7f8727e_2
#numpy                     1.21.4                   pypi_0    pypi
#openssl                   1.1.1l               h7f8727e_0
#pillow                    8.4.0                    pypi_0    pypi
#pip                       21.2.4           py38h06a4308_0
#pyparsing                 3.0.6                    pypi_0    pypi
#python                    3.8.12               h12debd9_0
#python-dateutil           2.8.2                    pypi_0    pypi
#python-graphviz           0.18.1                   pypi_0    pypi
#pyyaml                    6.0                      pypi_0    pypi
#readline                  8.1                  h27cfd23_0
#setuptools                58.0.4           py38h06a4308_0
#six                       1.16.0                   pypi_0    pypi
#sqlite                    3.36.0               hc218d9a_0
#tk                        8.6.11               h1ccaba5_0
#torch                     1.10.0                   pypi_0    pypi
#typing-extensions         4.0.0                    pypi_0    pypi
#wheel                     0.37.0             pyhd3eb1b0_1
#xz                        5.2.5                h7b6447c_0
#zlib                      1.2.11               h7b6447c_3

ADD . .

