package com.example.android.fyp;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {

    ImageView uploadImageView;
    Button addBtn;
    Button analyseButton;
    Button takePicButton;
    Button saveButton;
    TextView outputNumber;
    Uri selectedImage;
    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    Interpreter tflite;
    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    public static final int REQUEST_CAPTURE = 1;
    private static final int GALLERY_REQUEST_CODE = 100;
    Bitmap currImgBitmap;
    Boolean imagePicked = false;
    //String modelPath = "1.tflite";
    String modelPath = "1Data_aug_200x200_30epoch.tflite";//================================================================================
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    protected ByteBuffer imgData = null;
    /** Dimensions of inputs.===== should batch size be 1 or 32? */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;//===========================================================================
    private int numBytesPerChannel = 4;//32 bit float value needs 4 bytes
    //Benign & Malignant
    int numLabels = 1;
    int modelImgHeight = 200, modelImgWidth = 200;//=========================================================================
    //Holds label strings
    List<String> labelList = Arrays.asList("benign", "malignant");
    /** Number of results to show in the UI. */
    private static final int RESULTS_TO_SHOW = 2;//was 3
    /** multi-stage low pass filter * */
    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR = 0.4f;
    private float[][] filterLabelProbArray = new float[FILTER_STAGES][numLabels];
    private float[][] labelProbArray = new float[1][1];

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        uploadImageView = (ImageView) findViewById(R.id.uploadImageView);
        addBtn = (Button) findViewById(R.id.addBtn);
        takePicButton = (Button) findViewById(R.id.takePic);
        analyseButton = (Button) findViewById(R.id.analyseButton);
        analyseButton.setVisibility(View.GONE);
        outputNumber = (TextView) findViewById(R.id.outputNumber);
        outputNumber.setVisibility(View.GONE);
        saveButton = (Button) findViewById(R.id.saveButton);
        saveButton.setVisibility(View.GONE);

        addBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                pickFromGallery();
            }
        });
        takePicButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                launchCamera(view);
            }
        });

        try {
            tfliteModel = loadModelFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        //tflite = new Interpreter(tfliteModel, tfliteOptions);
        tflite = new Interpreter(tfliteModel);

        if(!hasCamera()){
            takePicButton.setEnabled(false);
        }

        analyseButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(imagePicked) {
                    float prediction = classify();
                    if(prediction == 0.0) {//esgfiobrwaibeauiob
                        outputNumber.setText("Result: Benign");
                    } else if(prediction == 1.0){
                        outputNumber.setText("Result: Malignant");
                    } else {
                        outputNumber.setText("Result: " + Float.toString(prediction));
                    }
                    analyseButton.setVisibility(View.GONE);
                    outputNumber.setVisibility(View.VISIBLE);
                    saveButton.setVisibility(View.VISIBLE);
                }
            }
        });
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    saveInGallery(currImgBitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private boolean hasCamera() {
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }

    private void launchCamera(View v){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if(resultCode == RESULT_OK) {
            if (requestCode == REQUEST_CAPTURE) {
                Bundle extras = data.getExtras();
                currImgBitmap= (Bitmap) extras.get("data");
                imagePicked = true;
                uploadImageView.setImageBitmap(Bitmap.createScaledBitmap(currImgBitmap, 200, 200, false));
            } else if (requestCode == GALLERY_REQUEST_CODE) {
                selectedImage = data.getData();
                InputStream imageStream = null;
                try {
                    imageStream = getContentResolver().openInputStream(selectedImage);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                currImgBitmap = BitmapFactory.decodeStream(imageStream);
                uploadImageView.setImageBitmap(Bitmap.createScaledBitmap(currImgBitmap, 200, 200, false));
                imagePicked = true;
            }
            outputNumber.setVisibility(View.VISIBLE);
            analyseButton.setVisibility(View.VISIBLE);
        }

    }

    private void pickFromGallery(){
        //Create an Intent with action as ACTION_PICK
        Intent intent=new Intent(Intent.ACTION_PICK);
        // Sets the type as image/*. This ensures only components of type image are selected
        intent.setType("image/*");
        //We pass an extra array with the accepted mime types. This will ensure only components with these MIME types as targeted.
        String[] mimeTypes = {"image/jpeg", "image/png"};
        intent.putExtra(Intent.EXTRA_MIME_TYPES,mimeTypes);
        // Launching the Intent
        startActivityForResult(intent,GALLERY_REQUEST_CODE);
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelPath);
        Log.i("error", "FD:" + fileDescriptor.toString());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public float classify() {

        //get Image height & Width
        Bitmap bOrig = ((BitmapDrawable) uploadImageView.getDrawable()).getBitmap();
        Bitmap b = Bitmap.createScaledBitmap(bOrig, modelImgWidth, modelImgHeight, true);
        int w = b.getWidth();
        int h = b.getHeight();

        if (tflite == null) {
            Log.e("error", "Image classifier has not been initialized; Skipped.");
        }

        convertBitmapToByteBuffer(b);

        long startTime = SystemClock.uptimeMillis();
        runInference();
        long endTime = SystemClock.uptimeMillis();
        Log.d("time", "Timecost to run model inference: " + Long.toString(endTime - startTime));
        Log.i("results", Arrays.deepToString(labelProbArray) + " ARRAY");
        //applyFilter();

        // Print the results.
        //printTopKLabels();


        long duration = endTime - startTime;
        Log.i("time", duration + " ms");

        return labelProbArray[0][0];
    }

    protected void runInference() {
        tflite.run(imgData, labelProbArray);
    }


    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        int imgWidth = bitmap.getWidth();
        int imgHeight = bitmap.getHeight();
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                                * imgWidth
                                * imgHeight
                                * DIM_PIXEL_SIZE
                                * numBytesPerChannel);

        imgData.order(ByteOrder.nativeOrder());
        Log.d("classifier", "Created a Tensorflow Lite Image Classifier.");

        imgData.rewind();
        int[] intValues = new int[imgWidth * imgHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        Log.i("p1", "ht: " + imgHeight +". wdt: "+imgWidth);
        Log.i("p1", "intValues length:  " + intValues.length);
        Log.i("p1", "imgData size:  " + DIM_BATCH_SIZE
                * imgWidth
                * imgHeight
                * DIM_PIXEL_SIZE
                * numBytesPerChannel);

        for (int i = 0; i < imgWidth; ++i) {
            for (int j = 0; j < imgHeight; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.i("bytebuffer", "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    protected void addPixelValue(int pixelValue) {
        //TODO look at getting rid of these
        final int IMAGE_MEAN = 128;
        final float IMAGE_STD = 128.0f;
        //!3 for when its color, and 1 for when its 2
       // imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        //imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        //TODO maybe put in just pixelValue here
        imgData.putFloat(pixelValue );
    }

    void applyFilter() {

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for (int j = 0; j < numLabels; ++j) {
            filterLabelProbArray[0][j] +=
                    FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
        }
        // Low pass filter each stage into the next.
        for (int i = 1; i < FILTER_STAGES; ++i) {
            for (int j = 0; j < numLabels; ++j) {
                filterLabelProbArray[i][j] +=
                        FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for (int j = 0; j < numLabels; ++j) {
            setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
        }
    }

    protected float getProbability(int labelIndex) {
        return labelProbArray[0][labelIndex];
    }

    protected void setProbability(int labelIndex, Number value) {
        labelProbArray[0][labelIndex] = value.floatValue();
    }

    protected float getNormalizedProbability(int labelIndex) {
        // TODO the following value isn't in [0,1] yet, but may be greater. Why?
        return getProbability(labelIndex);
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    private void printTopKLabels() {
        String test = labelList.get(0);
        Log.i("labels", "This should be benign: " + test);
        for (int i = 0; i < numLabels; ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        final int size = sortedLabels.size();
        for (int i = 0; i < size; i++) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            Log.i("results", String.format("%s: %4.2f\n", label.getKey(), label.getValue()));
        }
    }

    private void saveInGallery(Bitmap bitmap) throws IOException {

        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        int counter = 0;
        File imgToBeSaved;
        String NAME = "test";
        String NOMEDIA=".nomedia";
        File imgDirectory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        imgToBeSaved = new File(imgDirectory, NAME+NOMEDIA);

        while(imgToBeSaved.exists()){
            //iterate counter until we have a new filename
            counter++;
            imgToBeSaved = new File(imgDirectory, NAME+counter+NOMEDIA);
        }

        Log.i("storage", "RUNNING");
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(imgToBeSaved);
            Log.i("storage", "ABSOLUTE PATH:" + imgToBeSaved.getAbsolutePath());
            // Use the compress method on the BitMap object to write image to the OutputStream
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }




    }

    /*private String saveToInternalStorage(Bitmap bitmapImage){
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // path to /data/data/yourapp/app_data/imageDir
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir
        File mypath=new File(directory,"profile.jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            // Use the compress method on the BitMap object to write image to the OutputStream
            bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }
*/
}
