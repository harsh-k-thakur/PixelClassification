The code is easy to run! 

A. For pixel classification the main file is in the pixel _classification folder.
	"test_pixel_classifier.py" This is the file that you need to run.
	You can use "python test_pixel_classifier.py" to run this file
	The first part is it generate the rgb data. It takes the test data and convert it to [0, 1] range.
	Once this is done it creates an "myPixelClassifier" object.
	This is useful for training data or loading data.
	After loading the data it uses the "classify" method of "myPixelClassifier" object to classify the given pixel.
	And you recieve your  accuracy score.


B. For bin detection the main file is in the bin_detection folder.
	"test_bin_detector.py" This is the file that you need to run.
	You can use "python "test_bin_detector.py" to run this file
	The first part is to create an object called "my_detector".
	In this object you can label the data and according to your labels the model is trained and it is saved in a file. In order to label the data I am using roipoly. The same image is shown twice. For the first time you create a polygon around the object you want the color of. For the second time you draw a polygon that you don't want to select.
	For example my function has two labels blue and non-blue. For blue label, I make a polygon inside the dustbin. And for non-blue region, I make a polygon surrounding the dustbin. this will take all the pixel outside of the polygon as non-blue data.
	Once you have saved the model it reloads the model from that file.
	After that part, reading of image from the given folder is done.
	The image is passes inside the "segment_image" method of object called "my_detector".
	Here the image is segmented in binary format blue as white and non-blue as black.
	Then the segmented image is passed to the "get_bounding_boxes" method of object called "my_detector". 
	This method returns the bounding box and it is compared to the actual bounding box.
	And you recieve your accuracy score.















