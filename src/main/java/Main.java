import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    static {
        nu.pattern.OpenCV.loadLocally();
    }


    // to create a mask for a selected color range
    static Mat createMask(Mat hsv, Scalar lower, Scalar upper) {

        Mat mask = new Mat();

        Core.inRange(
                hsv,
                lower,
                upper,
                mask
        );

        // to remove small unwanted background noise
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                new Size(5, 5)
        );

        Imgproc.morphologyEx(
                mask,
                mask,
                Imgproc.MORPH_OPEN,
                kernel
        );

        Imgproc.morphologyEx(
                mask,
                mask,
                Imgproc.MORPH_CLOSE,
                kernel
        );

        kernel.release();

        return mask;
    }


    // to find the largest connected color area
    static double getLargestArea(Mat mask) {

        List<MatOfPoint> contours = new ArrayList<>();

        Mat hierarchy = new Mat();

        Imgproc.findContours(
                mask,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        double largestArea = 0;


        // to check each detected color area
        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            Rect box = Imgproc.boundingRect(contour);

            double ratio =
                    (double) box.width / Math.max(box.height, 1);


            // to keep regions that are roughly traffic light shaped
            if (area > 20
                    && ratio > 0.4
                    && ratio < 2.5) {

                if (area > largestArea) {
                    largestArea = area;
                }
            }

            contour.release();
        }


        hierarchy.release();

        return largestArea;
    }


    // to detect the traffic light color from the image
    static String getColor(Mat img) {

        Mat hsv = new Mat();


        // to convert image from BGR color format to HSV
        Imgproc.cvtColor(
                img,
                hsv,
                Imgproc.COLOR_BGR2HSV
        );


        /*
         * to mainly check the upper part of the image
         * because road traffic lights normally appear above the road
         */
        int roiHeight =
                Math.max(
                        1,
                        (int) (hsv.rows() * 0.85)
                );

        Rect roi =
                new Rect(
                        0,
                        0,
                        hsv.cols(),
                        roiHeight
                );

        Mat area =
                hsv.submat(roi);


        // to check the first red color range
        Mat redMask1 =
                createMask(
                        area,
                        new Scalar(0, 90, 120),
                        new Scalar(10, 255, 255)
                );


        // to check the second red color range
        Mat redMask2 =
                createMask(
                        area,
                        new Scalar(165, 90, 120),
                        new Scalar(180, 255, 255)
                );


        // to combine both red ranges
        Mat redMask =
                new Mat();

        Core.bitwise_or(
                redMask1,
                redMask2,
                redMask
        );


        // to check yellow or amber traffic light pixels
        Mat yellowMask =
                createMask(
                        area,
                        new Scalar(12, 70, 120),
                        new Scalar(38, 255, 255)
                );


        // to check green traffic light pixels
        Mat greenMask =
                createMask(
                        area,
                        new Scalar(40, 70, 100),
                        new Scalar(90, 255, 255)
                );


        // to find the strongest connected area for each color
        double red =
                getLargestArea(redMask);

        double yellow =
                getLargestArea(yellowMask);

        double green =
                getLargestArea(greenMask);


        // to release OpenCV memory
        redMask1.release();
        redMask2.release();
        redMask.release();
        yellowMask.release();
        greenMask.release();
        area.release();
        hsv.release();


        // to find the strongest detected traffic light color
        double best =
                Math.max(
                        red,
                        Math.max(yellow, green)
                );


        // to check if no clear traffic light color was detected
        if (best < 20) {
            return "UNKNOWN";
        }


        // to return the detected color
        if (best == red) {
            return "RED";
        }

        if (best == yellow) {
            return "YELLOW";
        }

        return "GREEN";
    }


    // to get the driving action based on detected color
    static String getAction(String color) {

        switch (color) {

            case "RED":
                return "STOP";

            case "YELLOW":
                return "SLOW DOWN";

            case "GREEN":
                return "GO";

            default:
                return "NO ACTION";
        }
    }


    // to get the expected color from the image file name
    static String getExpectedColor(String fileName) {

        String name =
                fileName.toLowerCase();

        if (name.contains("red")) {
            return "RED";
        }

        if (name.contains("yellow")) {
            return "YELLOW";
        }

        if (name.contains("green")) {
            return "GREEN";
        }

        return "UNKNOWN";
    }


    // main program to process all traffic light images
    public static void main(String[] args) {

        String folder =
                "sample_images";

        File dir =
                new File(folder);


        // to check if the sample image folder exists
        if (!dir.exists()) {

            System.out.println(
                    "sample_images folder was not found."
            );

            return;
        }


        // to collect png, jpg and jpeg image files
        File[] files =
                dir.listFiles(
                        (d, name) -> {

                            String lower =
                                    name.toLowerCase();

                            return lower.endsWith(".png")
                                    || lower.endsWith(".jpg")
                                    || lower.endsWith(".jpeg");
                        }
                );


        // to check if there are no images
        if (files == null
                || files.length == 0) {

            System.out.println(
                    "No images found in sample_images."
            );

            return;
        }


        // to sort image files by name
        Arrays.sort(files);


        System.out.println();
        System.out.println(
                "TRAFFIC LIGHT DETECTOR"
        );

        System.out.println(
                "----------------------"
        );


        int tested = 0;
        int correct = 0;


        // to process each image one by one
        for (File file : files) {

            Mat img = null;

            try {

                // to read the image using OpenCV
                img =
                        Imgcodecs.imread(
                                file.getPath()
                        );


                // to check if the image cannot be read
                if (img.empty()) {

                    System.out.println(
                            file.getName()
                                    + " -> could not read image"
                    );

                    continue;
                }


                // to detect traffic light color
                String color =
                        getColor(img);


                // to determine driving action
                String action =
                        getAction(color);


                // to get expected color from the file name
                String expected =
                        getExpectedColor(
                                file.getName()
                        );


                // to check whether classification is correct
                boolean isCorrect =
                        expected.equals(color);


                tested++;

                if (isCorrect) {
                    correct++;
                }


                // to print result
                System.out.println(
                        file.getName()
                                + " -> "
                                + color
                                + " -> "
                                + action
                                + " -> "
                                + (isCorrect ? "CORRECT" : "INCORRECT")
                );


            } catch (Exception e) {

                // to handle errors without stopping the whole program
                System.out.println(
                        file.getName()
                                + " -> error: "
                                + e.getMessage()
                );


            } finally {

                // to release image memory
                if (img != null) {
                    img.release();
                }
            }
        }


        // to calculate testing accuracy
        double accuracy = 0;

        if (tested > 0) {

            accuracy =
                    ((double) correct / tested) * 100;
        }


        System.out.println();
        System.out.println(
                "Images tested: " + tested
        );

        System.out.println(
                "Correct detections: "
                        + correct
                        + "/"
                        + tested
        );

        System.out.printf(
                "Accuracy: %.1f%%%n",
                accuracy
        );


        System.out.println();
        System.out.println(
                "Processing completed."
        );
    }
}