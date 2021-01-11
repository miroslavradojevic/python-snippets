close all; clear all; clc;

N = 256;

C = jet(N);

C = C * 255; % byte gray8 image values

C = round(C);

C(C>255)    = 255;
C(C<0  )    = 0;

fprintf(1, 'int[][] jet = new int[][]{\n');
for i = 1 : 256,
    if(i==256),
        fprintf(1, '{%4d, %4d, %4d}\n', C(i, 1), C(i, 2), C(i, 3));
        continue;
    end
    fprintf(1, '{%4d, %4d, %4d}, ', C(i, 1), C(i, 2), C(i, 3));
    if(mod(i,5)==0),
        fprintf(1, '\n');
    end
end
fprintf(1, '};\n');

