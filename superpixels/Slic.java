package superpixels.core;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import advantra.tools.Find_Connected_Regions;

import ij.process.ByteProcessor;
import ij.process.ColorProcessor;

import ij.ImagePlus;
import ij.gui.NewImage;

public class Slic {
	
	/*
	 * based on:
	 * Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, 
	 * SLIC Superpixels, EPFL Technical Report no. 149300, June 2010.
	 *
	 * Achanta, Radhakrishna, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk. 
	 * "SLIC superpixels compared to state-of-the-art superpixel methods." (2012): 1-1.
	 * IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 34, NO. 11, NOVEMBER 2012
	 */

	ImagePlus 	img;
	
	int 		K;
	float[][]	C; 			// clusters (cluster data is stored in each row)
	// row: 1-grey level 2-row (marked as x or h) 3-col (y or w) 4-lay (marked as l or z) 5-size
	float[][] 	img_array;	// stores vales for the image voxels
	
	int[][][]	lab; 		// label per each 3d voxel
	float[][][]	dist;		// normalized distance
	
	int 		S1;		 	// grid interval (layer)
	int			S2;			// grid interval (for layers' dimension)
	int 		L;			// layers nr.
	int 		H;			// height
	int 		W;			// width
	
	public Slic(ImagePlus img, int K){
		
		this.img 	= img;
		this.K 		= K;
		
		H 	= img.getHeight();
		W 	= img.getWidth();
		L 	= img.getStack().getSize();
		
		img_array = new float[L][H*W];
		for (int l = 0; l < L; l++) {
			byte[] pix = (byte [])img.getStack().getProcessor(l+1).getPixels();
			for (int i = 0; i < pix.length; i++) {
				img_array[l][i] = (int)(pix[i]&0xff);
			}
		}
		
		lab = new int[H][W][L];
		
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				for (int k = 0; k < L; k++) {
					lab[i][j][k] = this.K*this.L; 		// set the initial label to L*K - the rest will be filled from 0-(K-1)
				}
			}
		}		
		
		dist= new float[H][W][L];
		
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				for (int k = 0; k < L; k++) {
					dist[i][j][k] = Float.MAX_VALUE;				// set the distances to max possible value
				}
			}
		}
		
		C	= new float[K*L][5];
		
		S1 = (int)Math.ceil( Math.pow((double)(H*W)/this.K, 0.5) ); // grid interval in 2d
		
		int idx = 0;
		
		for (int l = 0; l < L; l++) {
			
			// sample K regularly spaced cluster centers per layer
			int cnt_per_layer = 0;
			for (int i = 0; (i<H)&&(cnt_per_layer<this.K); i+=S1) {
				for (int j = 0; (j<W)&&(cnt_per_layer<this.K); j+=S1) {
					C[idx][0] = img_array[l][i*W+j]; 
					C[idx][1] = i;
					C[idx][2] = j;
					C[idx][3] = l;
					C[idx][4] = 1;
					idx++;
					cnt_per_layer++;
				}
			}
			
			// move them to lowest gradient position within 3x3 neighborhood
			for(int i = l*this.K; i < (l+1)*this.K; i++){ // take K previous clusters
				
				int curr_x = (int)Math.round(C[i][1]);
				int curr_y = (int)Math.round(C[i][2]);
				
				float min_grad = Float.MAX_VALUE;
				
				for(int nx = curr_x-1; nx <= curr_x+1; nx++){
					for(int ny = curr_y-1; ny <= curr_y+1; ny++){
						// around (nx, ny)
						if((nx-1)>=0 && (nx+1)<H && (ny-1)>=0 && (ny+1)<W){
							float I12 = img_array[l][(nx+1)*W+ ny   ];
							float I11 = img_array[l][(nx-1)*W+ ny   ];
							float I22 = img_array[l][ nx   *W+(ny+1)];
							float I21 = img_array[l][ nx   *W+(ny-1)];
							float curr_grad = Math.abs(I12-I11)+Math.abs(I22-I21);
								
							if(curr_grad<min_grad){
								C[i][0] = img_array[l][nx*W+ny];
								C[i][1] = nx;
								C[i][2] = ny;
								min_grad = curr_grad;
							}
						}
					}
				}
			}
		}
	}		
	
	public void run(int max_iter, float m){
		
		long t_total = System.currentTimeMillis();
				
		for (int l = 0; l < L; l++) {
					
					System.out.print("\nlayer "+(l)+" / "+(L-1)+", "+max_iter+" iter : ");
					long t_layer = System.currentTimeMillis();
					
					int iter = 0;
					
					while(iter<max_iter){ 
						
						System.out.print(" "+iter+" ");
						
						for (int i = l*K; i < (l+1)*K; i++) { // loop clusters
							
							int x_beg 	= (int)Math.round(C[i][1]-S1);
							int x_end 	= (int)Math.round(C[i][1]+S1);
							int y_beg 	= (int)Math.round(C[i][2]-S1);
							int y_end 	= (int)Math.round(C[i][2]+S1);
							
							for (int x = x_beg; x <= x_end; x++) {
								for (int y = y_beg; y <= y_end; y++) {
									if(x>=0 && x<H && y>=0 && y<W){
										
										float dh 	= x-C[i][1];
										float dw 	= y-C[i][2];
										float dg 	= img_array[l][x*W+y]-C[i][0];
										float D 	= dg*dg + (float)(Math.pow((m/(float)S1), 2))*(dh*dh + dw*dw); 
										
										if(D<dist[x][y][l]){
											dist[x][y][l] 	= D;
											lab[x][y][l]	= i;
										}
									}
								}
							}
						}
						
						// discard those that were not connected to the main of the region (this part takes time...)
						Find_Connected_Regions find_conn = new Find_Connected_Regions();
						for (int i = l*K; i < (l+1)*K; i++) { // loop clusters
							// check per cluster
							int patch_size = 2*S1+1;
							byte[] newImg_array = new byte[patch_size*patch_size];
							
							int x_beg 	= (int)Math.round(C[i][1]-S1);
							int x_end 	= (int)Math.round(C[i][1]+S1);
							int y_beg 	= (int)Math.round(C[i][2]-S1);
							int y_end 	= (int)Math.round(C[i][2]+S1);
							
							for (int x = x_beg; x <= x_end; x++) {
								for (int y = y_beg; y <= y_end; y++) {
									if(x>=0 && x<H && y>=0 && y<W) {
										
										int patch_row = x-x_beg;
										int patch_col = y-y_beg;
										
										boolean isFromThisCluster = lab[x][y][l]==i;
										
										if(isFromThisCluster){
											newImg_array[patch_row*patch_size+patch_col] = (byte)255;
										}
										else{
											newImg_array[patch_row*patch_size+patch_col] = (byte)0;
										}
										
									}
								}
							}
							
							ImagePlus newImg = new ImagePlus("", new ByteProcessor(patch_size, patch_size, newImg_array));
							
							// check if it's connected to the body
							find_conn.set(newImg, true);
							
							find_conn.run("");
							
							for (int x = x_beg; x <= x_end; x++) {
								for (int y = y_beg; y <= y_end; y++) {
									if(x>=0 && x<H && y>=0 && y<W) {
										int patch_row = x-x_beg;
										int patch_col = y-y_beg;
										int[] cp = new int[]{patch_row, patch_col, 0};
										
										if(lab[x][y][l]==i){
											if(!find_conn.belongsToBiggestRegion(cp)){
												// cancel the label & weights
												lab[x][y][l] 	= this.K*this.L;
												dist[x][y][l] 	= Float.MAX_VALUE;												
											}
										}
									}
								}
							}
						}
						
						for (int i = l*K; i < (l+1)*K; i++) { // update the cluster_centers 
							
							int x_beg 	= (int)Math.round(C[i][1]-S1);
							int x_end 	= (int)Math.round(C[i][1]+S1);
							int y_beg 	= (int)Math.round(C[i][2]-S1);
							int y_end 	= (int)Math.round(C[i][2]+S1);
							
							float 	new_value 	= 0;
							float 	new_row   	= 0;
							float 	new_col   	= 0;
							int 	cnt 		= 0;
							
							for (int x = x_beg; x <= x_end; x++) {
								for (int y = y_beg; y <= y_end; y++) {
									if(x>=0 && x<H && y>=0 && y<W && lab[x][y][l]==i){
										new_value 	+= img_array[l][x*W+y];
										new_row 	+= x;
										new_col 	+= y;
										cnt++;
									}
								}
							}
							if(cnt>0){
								
								new_value 	/= cnt;
								new_row 	/= cnt;
								new_col 	/= cnt;
								
								C[i][0] = new_value;
								C[i][1] = new_row;
								C[i][2] = new_col;
								C[i][4] = cnt;
								
							}
							
						}
						
						iter++;
						
					} // iterations for the cluster updates over
					
					System.out.println("done. elapsed: "+(System.currentTimeMillis()-t_layer)/1000f+" sec.");
					
					System.out.println("eliminate orphaned pixels...");
					
					//int orphaned_cnt = Integer.MAX_VALUE;
					//orphaned_cnt = 0;
					for (int row = 0; row < H; row++) {
						for (int col = 0; col < W; col++) {
							if(lab[row][col][l]==this.K*this.L){
								float min_dist = Float.MAX_VALUE;
								int[][] nbrs = get4Neighbors(row, col);
								for (int j = 0; j < nbrs.length; j++) {
									if(lab[nbrs[j][0]][nbrs[j][1]][l]!=K*L){
										int c_idx = lab[nbrs[j][0]][nbrs[j][1]][l];
										float dh 	= row - C[c_idx][1];
										float dw 	= col - C[c_idx][2];
										float dg 	= img_array[l][row*W+col]-C[c_idx][0];
										float D 	= dg*dg + (float)(Math.pow((m/(float)S1), 2))*(dh*dh + dw*dw); 
										if(D<min_dist){
											min_dist = D;
											lab[row][col][l] = c_idx;
										}
									}
									
								}

							}
						}
					}
					
		} // layer done 
		System.out.println("total elapsed: "+(System.currentTimeMillis()-t_total)/1000f+" sec.");
				
	}
	
	/* not used in current implementation
	private int[][] get8Neighbors(int pos_x, int pos_y){
		
		int cnt_nbr = 0;
		
		for (int i = pos_x-1; i <= pos_x+1; i++) {
			for (int j = pos_y-1; j <= pos_y+1; j++) {
				boolean isInImage = i>=0 && i<H && j>=0 && j<W;
				if(!(i==pos_x && j==pos_y) && isInImage){
					cnt_nbr++;
				}
			}
		}
		
		int[][] nbrs = new int[cnt_nbr][2];
		
		cnt_nbr = 0;
		
		for (int i = pos_x-1; i <= pos_x+1; i++) {
			for (int j = pos_y-1; j <= pos_y+1; j++) {
				boolean isInImage = i>=0 && i<H && j>=0 && j<W;
				if(!(i==pos_x && j==pos_y) && isInImage){
					nbrs[cnt_nbr][0] = i;
					nbrs[cnt_nbr][1] = j;
					cnt_nbr++;
				}
			}
		}
		
		return nbrs;
	}
	*/
	
	private int[][] get4Neighbors(int pos_x, int pos_y){
		
		int[][] conn4 = new int[4][2];
		conn4[0][0] = pos_x-1; 	conn4[0][1] = pos_y; 
		conn4[1][0] = pos_x+1; 	conn4[1][1] = pos_y;
		conn4[2][0] = pos_x; 	conn4[2][1] = pos_y-1;
		conn4[3][0] = pos_x; 	conn4[3][1] = pos_y+1;
		
		int cnt_nbr = 0;
		
		for (int i = 0; i < 4; i++) {
			boolean isInImage = conn4[i][0]>=0 && conn4[i][0]<H && conn4[i][1]>=0 && conn4[i][1]<W;
			if(isInImage){
				cnt_nbr++;
			}
		}
		
		int[][] nbrs = new int[cnt_nbr][2];
		
		cnt_nbr = 0;
		
		for (int i = 0; i < 4; i++) {
			boolean isInImage = conn4[i][0]>=0 && conn4[i][0]<H && conn4[i][1]>=0 && conn4[i][1]<W;
			if(isInImage){
				nbrs[cnt_nbr][0] = conn4[i][0];
				nbrs[cnt_nbr][1] = conn4[i][1];
				cnt_nbr++;
			}
		}
		
		return nbrs;
	}
	
	public float[][] extract_cluster_centers(){
		
		float[][] cent = new float[K][3];
		int idx = 0;
		for (int i = 0; i < K*L; i++) {
			if(C[i][0]>0){
				cent[idx][0] = C[i][1];
				cent[idx][1] = C[i][2];
				cent[idx][2] = C[i][3];
				idx++;
			}
		}
		
		return cent;
		
	}
	
	public String export_cluster_centers(){
		
		String file_name = img.getTitle()+"_clusters.csv";
		FileWriter fw; 
		
		try{
			fw = new FileWriter(file_name);
			fw.write(String.format("# superpixel locations, input image: %s\n", img.getTitle()));
			fw.write(String.format("# col,\trow,\tlay \n"));
			for (int i = 0; i < K*L; i++) {
				if(C[i][0]>0){
					fw.write(String.format("%5.2f, %5.2f, %5.2f\n", C[i][2], C[i][1], C[i][3]));
				}
			}
			fw.close();
		} 
		catch(IOException exIO){}
		
		System.out.println(file_name+" exported!");
		
		return file_name;
		
	}
	
	public ImagePlus outputLabelled(){
		
		ImagePlus out = NewImage.createRGBImage("Labels", W, H, L, NewImage.FILL_WHITE);
		
		// make a lookup table
		byte[][] table = new byte[L*K+1][3];
		for (int i = 0; i < L*K; i++) {
			table[i] = random_jet256();
		}
		table[L*K] = new byte[]{(byte)0, (byte)0, (byte)0};
		
		byte[][][]  img_rgb_stack = new byte[3][L][W*H];

    	for (int row = 0; row < H; row++) {
    		for (int col = 0; col < W; col++) {
				for (int lay = 0; lay < L; lay++) {
					int cluster_idx = lab[row][col][lay];
					if(cluster_idx>=0 && cluster_idx<=K*L){ // && C[cluster_idx][0]>0
						byte r = table[cluster_idx][0];
						byte g = table[cluster_idx][1];
						byte b = table[cluster_idx][2];
						img_rgb_stack[0][lay][row*W+col] = r;
			    		img_rgb_stack[1][lay][row*W+col] = g;
			    		img_rgb_stack[2][lay][row*W+col] = b;
					}
				}
			}
		}
		
		for (int i = 1; i <= out.getStack().getSize(); i++) {
			((ColorProcessor)out.getStack().getProcessor(i)).setRGB(
					img_rgb_stack[0][i-1], 
					img_rgb_stack[1][i-1], 
					img_rgb_stack[2][i-1]
			);
		}

		return out;
	}
	
	private static byte[] random_jet256(){
		
		Random generator = new Random();
		int choose_random_color = generator.nextInt(256);

		int[][] jet256 = new int[][]{
				{   0,    0,  131}, {   0,    0,  135}, {   0,    0,  139}, {   0,    0,  143}, {   0,    0,  147}, 
				{   0,    0,  151}, {   0,    0,  155}, {   0,    0,  159}, {   0,    0,  163}, {   0,    0,  167}, 
				{   0,    0,  171}, {   0,    0,  175}, {   0,    0,  179}, {   0,    0,  183}, {   0,    0,  187}, 
				{   0,    0,  191}, {   0,    0,  195}, {   0,    0,  199}, {   0,    0,  203}, {   0,    0,  207}, 
				{   0,    0,  211}, {   0,    0,  215}, {   0,    0,  219}, {   0,    0,  223}, {   0,    0,  227}, 
				{   0,    0,  231}, {   0,    0,  235}, {   0,    0,  239}, {   0,    0,  243}, {   0,    0,  247}, 
				{   0,    0,  251}, {   0,    0,  255}, {   0,    4,  255}, {   0,    8,  255}, {   0,   12,  255}, 
				{   0,   16,  255}, {   0,   20,  255}, {   0,   24,  255}, {   0,   28,  255}, {   0,   32,  255}, 
				{   0,   36,  255}, {   0,   40,  255}, {   0,   44,  255}, {   0,   48,  255}, {   0,   52,  255}, 
				{   0,   56,  255}, {   0,   60,  255}, {   0,   64,  255}, {   0,   68,  255}, {   0,   72,  255}, 
				{   0,   76,  255}, {   0,   80,  255}, {   0,   84,  255}, {   0,   88,  255}, {   0,   92,  255}, 
				{   0,   96,  255}, {   0,  100,  255}, {   0,  104,  255}, {   0,  108,  255}, {   0,  112,  255}, 
				{   0,  116,  255}, {   0,  120,  255}, {   0,  124,  255}, {   0,  128,  255}, {   0,  131,  255}, 
				{   0,  135,  255}, {   0,  139,  255}, {   0,  143,  255}, {   0,  147,  255}, {   0,  151,  255}, 
				{   0,  155,  255}, {   0,  159,  255}, {   0,  163,  255}, {   0,  167,  255}, {   0,  171,  255}, 
				{   0,  175,  255}, {   0,  179,  255}, {   0,  183,  255}, {   0,  187,  255}, {   0,  191,  255}, 
				{   0,  195,  255}, {   0,  199,  255}, {   0,  203,  255}, {   0,  207,  255}, {   0,  211,  255}, 
				{   0,  215,  255}, {   0,  219,  255}, {   0,  223,  255}, {   0,  227,  255}, {   0,  231,  255}, 
				{   0,  235,  255}, {   0,  239,  255}, {   0,  243,  255}, {   0,  247,  255}, {   0,  251,  255}, 
				{   0,  255,  255}, {   4,  255,  251}, {   8,  255,  247}, {  12,  255,  243}, {  16,  255,  239}, 
				{  20,  255,  235}, {  24,  255,  231}, {  28,  255,  227}, {  32,  255,  223}, {  36,  255,  219}, 
				{  40,  255,  215}, {  44,  255,  211}, {  48,  255,  207}, {  52,  255,  203}, {  56,  255,  199}, 
				{  60,  255,  195}, {  64,  255,  191}, {  68,  255,  187}, {  72,  255,  183}, {  76,  255,  179}, 
				{  80,  255,  175}, {  84,  255,  171}, {  88,  255,  167}, {  92,  255,  163}, {  96,  255,  159}, 
				{ 100,  255,  155}, { 104,  255,  151}, { 108,  255,  147}, { 112,  255,  143}, { 116,  255,  139}, 
				{ 120,  255,  135}, { 124,  255,  131}, { 128,  255,  128}, { 131,  255,  124}, { 135,  255,  120}, 
				{ 139,  255,  116}, { 143,  255,  112}, { 147,  255,  108}, { 151,  255,  104}, { 155,  255,  100}, 
				{ 159,  255,   96}, { 163,  255,   92}, { 167,  255,   88}, { 171,  255,   84}, { 175,  255,   80}, 
				{ 179,  255,   76}, { 183,  255,   72}, { 187,  255,   68}, { 191,  255,   64}, { 195,  255,   60}, 
				{ 199,  255,   56}, { 203,  255,   52}, { 207,  255,   48}, { 211,  255,   44}, { 215,  255,   40}, 
				{ 219,  255,   36}, { 223,  255,   32}, { 227,  255,   28}, { 231,  255,   24}, { 235,  255,   20}, 
				{ 239,  255,   16}, { 243,  255,   12}, { 247,  255,    8}, { 251,  255,    4}, { 255,  255,    0}, 
				{ 255,  251,    0}, { 255,  247,    0}, { 255,  243,    0}, { 255,  239,    0}, { 255,  235,    0}, 
				{ 255,  231,    0}, { 255,  227,    0}, { 255,  223,    0}, { 255,  219,    0}, { 255,  215,    0}, 
				{ 255,  211,    0}, { 255,  207,    0}, { 255,  203,    0}, { 255,  199,    0}, { 255,  195,    0}, 
				{ 255,  191,    0}, { 255,  187,    0}, { 255,  183,    0}, { 255,  179,    0}, { 255,  175,    0}, 
				{ 255,  171,    0}, { 255,  167,    0}, { 255,  163,    0}, { 255,  159,    0}, { 255,  155,    0}, 
				{ 255,  151,    0}, { 255,  147,    0}, { 255,  143,    0}, { 255,  139,    0}, { 255,  135,    0}, 
				{ 255,  131,    0}, { 255,  128,    0}, { 255,  124,    0}, { 255,  120,    0}, { 255,  116,    0}, 
				{ 255,  112,    0}, { 255,  108,    0}, { 255,  104,    0}, { 255,  100,    0}, { 255,   96,    0}, 
				{ 255,   92,    0}, { 255,   88,    0}, { 255,   84,    0}, { 255,   80,    0}, { 255,   76,    0}, 
				{ 255,   72,    0}, { 255,   68,    0}, { 255,   64,    0}, { 255,   60,    0}, { 255,   56,    0}, 
				{ 255,   52,    0}, { 255,   48,    0}, { 255,   44,    0}, { 255,   40,    0}, { 255,   36,    0}, 
				{ 255,   32,    0}, { 255,   28,    0}, { 255,   24,    0}, { 255,   20,    0}, { 255,   16,    0}, 
				{ 255,   12,    0}, { 255,    8,    0}, { 255,    4,    0}, { 255,    0,    0}, { 251,    0,    0}, 
				{ 247,    0,    0}, { 243,    0,    0}, { 239,    0,    0}, { 235,    0,    0}, { 231,    0,    0}, 
				{ 227,    0,    0}, { 223,    0,    0}, { 219,    0,    0}, { 215,    0,    0}, { 211,    0,    0}, 
				{ 207,    0,    0}, { 203,    0,    0}, { 199,    0,    0}, { 195,    0,    0}, { 191,    0,    0}, 
				{ 187,    0,    0}, { 183,    0,    0}, { 179,    0,    0}, { 175,    0,    0}, { 171,    0,    0}, 
				{ 167,    0,    0}, { 163,    0,    0}, { 159,    0,    0}, { 155,    0,    0}, { 151,    0,    0}, 
				{ 147,    0,    0}, { 143,    0,    0}, { 139,    0,    0}, { 135,    0,    0}, { 131,    0,    0}, 
				{ 128,    0,    0}
				};
		
		//System.out.println("jet length: "+jet256.length);
		
		byte[] rgb = new byte[]{
				(byte)(jet256[choose_random_color][0]),
				(byte)(jet256[choose_random_color][1]),
				(byte)(jet256[choose_random_color][2])
				};
		
		return rgb;
	}


}
