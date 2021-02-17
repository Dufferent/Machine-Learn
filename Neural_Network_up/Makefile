PRO  = Neural_Network
OBJ  = demo.o neural.o
CC   = g++
MYFLAG = -L/home/xny/Downloads/opencv-3.4.12/x64_out/lib \
		 -Wl,-rpath=/home/xny/Downloads/opencv-3.4.12/x64_out/lib -lopencv_world \
		 -I/home/xny/Downloads/opencv-3.4.12/x64_out/include \
		 -I/home/xny/Downloads/opencv-3.4.12/x64_out/include/opencv \
		 -I/home/xny/Downloads/opencv-3.4.12/x64_out/include/opencv2 \
	     

$(PRO) : $(OBJ)
	$(CC) -o $@ $^ $(MYFLAG)

%.o : %.cpp
	$(CC) -o $@ -c $< $(MYFLAG)

clean:
	rm -rf $(PRO) $(OBJ)