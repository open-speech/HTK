# HTK 

If you don't know HTK, see this: [What is HTK?](http://htk.eng.cam.ac.uk/)

If you know HTK, you may wonder why this repo appears? Here are reasons:

- Though many great toolkits appears after HTK, it still has advantages for some applications.
  - less dependencies
  - less footprint 
- There is no an actively maintained HTK repo.
- However, there are some issues need to be deal with to use HTK today.



## Some changs

- add HDecode
- add HTKBook
- fix `make` issuse
- fix lexicon size limit
- fix some stack overflow errors, due to unreasonable buffer length


## Usage

The branch r3.4.1_fix contains the changes. To compile HTK:
```bash
git clone https://github.com/open-speech/HTK.git
cd HTK
./configure --prefix=$PWD --disable-hslab
make all
make install
# continue make HDecode
make hdecode
make install-hdecode
```


## Contributions

Contributions are welcome for this repo!


## Todo

- [ ] Give some examples or environments about HTK usages, for ASR e.g..