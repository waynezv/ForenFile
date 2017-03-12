VPATH = .

arch = linux
INSTD = ./
OBJPATH = ./
SRCPATH = ./

linux_CFLAGS = -g -Dintel -Dint32=int -Dint16=short \
		-Dfloat32=float -Dfloat64=double -static

alpha_osf1_CC = gcc
linux_CC = gcc


CFLAGS = ${$(arch)_CFLAGS} \
	-I$(VPATH) 




# Fri Mar 10 18:06:15 EST 1995 by PJM:
#
# make sure you call the libs in this order (1st sp, 2nd util_nist) , 
# otherwise it fails!!

LIBS = -lm 
TARGET = wave2feat
#test_newfe
OBJS  = 	\
	new_fe_wrapper.o new_fe.o new_fe_sp.o
default:
	( $(MAKE) -f $(VPATH)/Makefile CC=${$(arch)_CC}  $(TARGET))


$(TARGET): $(OBJS) 
	$(CC) -o $@ $(CFLAGS) $(OBJS) $(LIBS)
#	mv $(TARGET) $arch/$(TARGET)


clean:
	rm -f *.o core  *.c.*

#.c.o:
#	$(CC) -c -o $@ $< $(CFLAGS)

