package superpixels.main;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;

import java.io.File;

import superpixels.core.Slic;

public class Run implements PlugInFilter {

	ImagePlus 	_img; // because setup() and run( have to exchange global variable)
	int			_K 			= 500;
	int			_max_iter 	= 10;
	int			_m 			= 20;
	
	public static void main(String[] args){
		
		String img_name 	= "";
		int K 				= -1;
		int max_iter 		= 10;
		float m				= 20f;
		
		if(args.length==4){
			img_name 		= args[0];
			K 				= Integer.parseInt(args[1]);
			max_iter 		= Integer.parseInt(args[2]);
			m 				= Float.parseFloat(args[3]);
		}
		else if(args.length==3){
			img_name 		= args[0];
			K 				= Integer.parseInt(args[1]);
			m 				= Float.parseFloat(args[2]);
		}
		else if(args.length==2){
			img_name 		= args[0];
			K 				= Integer.parseInt(args[1]);			
		}
		else{
			System.out.println(
					"======================================================\n" +
					"Wrong number of command-line parameters.\n" +
					"******************************\n" +
					"USAGE INSTRUCTIONS\n" +
					"******************************\n" +
					"java -jar Superpixels.java INFILE	CLUSTER_NR	MAX_ITER	GRAY_NORM \n" +
					"java -jar Superpixels.java INFILE	CLUSTER_NR	GRAY_NORM \n" +
					"java -jar Superpixels.java INFILE	CLUSTER_NR	\n"
					);
			return;
		}
		
		if((new File(img_name)).exists()){
			img_name = (new File(img_name)).getAbsolutePath();
		}
		else{
			System.out.println("file "+img_name+" does not exist!");
			return;
		}
		
		ImagePlus img = new ImagePlus(img_name);
		
		Slic slic_superpix  = new Slic(img, K);
		slic_superpix.run(max_iter, m);
		
		slic_superpix.export_cluster_centers();
		//slic_superpix.export_labels();
		//slic_superpix.extract_cluster_centers();
		
		
	}

	public void run(ImageProcessor arg0) {
		GenericDialog gd = new GenericDialog("Superpixels", IJ.getInstance());
		gd.addNumericField("superpix. number   :", _K, 			0, 5, "");
		gd.addNumericField("max iterations     :", _max_iter,  	0, 5, "");
		gd.addNumericField("gray normalization :", _m, 	 		0, 5, "");  
		gd.showDialog();
		if (gd.wasCanceled()) return;
		_K 			= (int)gd.getNextNumber();
		_max_iter  	= (int)gd.getNextNumber();
		_m   		= (int)gd.getNextNumber();
		
		Slic slic_superpix  = new Slic(_img, _K);
		slic_superpix.run(_max_iter, _m);
		
		ImagePlus im_lab = slic_superpix.outputLabelled();
		im_lab.setTitle("Superpixels_in_random_colors");
		im_lab.show();
		
		String exported_file_name = slic_superpix.export_cluster_centers();
		exported_file_name = new File(exported_file_name).getAbsolutePath();
		IJ.log(exported_file_name);
		
	}

	public int setup(String arg0, ImagePlus img) {
		this._img = img;
		return DOES_8G+NO_CHANGES;
	}
	
}
