// app/lab-04/page.js
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { BookOpen, Code, CheckCircle, ScanEye } from "lucide-react";

export default function Lab04() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">Lab 04: Feature Extraction</h1>
        <p className="text-slate-600 text-lg">
          HOG, LBP, filtering, convolution, edge detection, and corner detection
          techniques
        </p>
      </div>

      <Tabs defaultValue="concepts" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="concepts">
            <BookOpen className="w-4 h-4 mr-2" />
            Concepts
          </TabsTrigger>
          <TabsTrigger value="code">
            <Code className="w-4 h-4 mr-2" />
            Code Examples
          </TabsTrigger>
          <TabsTrigger value="tasks">
            <CheckCircle className="w-4 h-4 mr-2" />
            Tasks
          </TabsTrigger>
        </TabsList>

        <TabsContent value="concepts" className="space-y-6 mt-6">
          <ConceptCard
            title="Introduction to Feature Extraction"
            icon={ScanEye}
          >
            <div className="space-y-4 text-slate-700">
              <p>
                Feature extraction is a fundamental technique in computer vision
                that focuses on selecting and transforming relevant information
                from raw visual data to create a more compact and meaningful
                representation. This process reduces dimensionality while
                enhancing relevant information.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  Why Feature Extraction?
                </p>
                <ul className="space-y-2 text-blue-800">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      Raw visual data is too complex and high-dimensional
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Reduces computational requirements</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      Captures essential information about objects and patterns
                    </span>
                  </li>
                </ul>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mt-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Local Features
                  </h4>
                  <p className="text-sm text-green-800">
                    Extracted from specific regions (keypoints, corners,
                    patches). Used for image matching and object detection.
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Global Features
                  </h4>
                  <p className="text-sm text-purple-800">
                    Computed over entire image (color histograms, texture
                    descriptors). Capture overall characteristics.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Feature Extraction Techniques">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="hog">
                <AccordionTrigger>
                  HOG (Histogram of Oriented Gradients)
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700">
                    <p>
                      Captures information about the distribution of gradient
                      directions in an image. Particularly useful for object
                      detection and pedestrian detection.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Working Steps:</p>
                      <ol className="space-y-1 text-sm ml-4 list-decimal">
                        <li>Convert image to grayscale</li>
                        <li>
                          Calculate gradient magnitude and orientation (using
                          Sobel)
                        </li>
                        <li>Divide image into cells (e.g., 8×8 pixels)</li>
                        <li>
                          Create histograms of gradient orientations for each
                          cell
                        </li>
                        <li>Group cells into blocks (e.g., 2×2 cells)</li>
                        <li>Normalize histograms within blocks</li>
                        <li>
                          Concatenate normalized histograms to form feature
                          vector
                        </li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="lbp">
                <AccordionTrigger>LBP (Local Binary Pattern)</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700">
                    <p>
                      Texture feature extraction technique that captures local
                      patterns by comparing pixel intensity with neighboring
                      pixels. Excellent for texture classification and face
                      recognition.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Working Steps:</p>
                      <ol className="space-y-1 text-sm ml-4 list-decimal">
                        <li>Define circular neighborhood around each pixel</li>
                        <li>
                          Compare center pixel intensity with each neighbor
                        </li>
                        <li>Assign 1 if neighbor ≥ center, else 0</li>
                        <li>Concatenate binary values to form LBP pattern</li>
                        <li>
                          Create histogram of LBP patterns across entire image
                        </li>
                        <li>Use histogram as feature vector</li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="hed">
                <AccordionTrigger>
                  HED (Histogram of Edge Directions)
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700">
                    <p>
                      Captures the distribution of edge orientations in an
                      image. Useful for texture analysis, object recognition,
                      and image segmentation.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Working Steps:</p>
                      <ol className="space-y-1 text-sm ml-4 list-decimal">
                        <li>Apply Canny edge detection to get edge map</li>
                        <li>Calculate gradient direction at each edge pixel</li>
                        <li>
                          Divide orientations into bins (e.g., 8 bins for 360°)
                        </li>
                        <li>Assign each pixel's orientation to a bin</li>
                        <li>
                          Create histogram representing edge direction
                          distribution
                        </li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="hig">
                <AccordionTrigger>
                  HIG (Histogram of Intensity Gradients)
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700">
                    <p>
                      Captures information about the distribution of intensity
                      gradients, representing how pixel intensities change
                      across the image.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Working Steps:</p>
                      <ol className="space-y-1 text-sm ml-4 list-decimal">
                        <li>Compute gradient using Sobel operator (dx, dy)</li>
                        <li>Calculate magnitude: G = sqrt(dx² + dy²)</li>
                        <li>Calculate orientation: θ = atan2(dy, dx)</li>
                        <li>Create histogram of gradient orientations</li>
                        <li>Optionally normalize histogram</li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </ConceptCard>

          <ConceptCard title="Filtering & Convolution">
            <div className="space-y-4 text-slate-700">
              <p>
                Filtering involves applying a kernel (small matrix) to an image
                to modify pixel values. Convolution is the mathematical
                operation used to apply filters, involving sliding the kernel
                over the image and computing dot products.
              </p>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">Box Blur</h4>
                  <p className="text-sm text-blue-800">
                    Simple averaging of neighboring pixels within a square
                    kernel. Good for basic noise reduction.
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Gaussian Blur
                  </h4>
                  <p className="text-sm text-green-800">
                    Uses Gaussian-shaped kernel for smoother results. Preferred
                    for noise reduction.
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Edge Detection
                  </h4>
                  <p className="text-sm text-purple-800">
                    Sobel and Scharr operators detect edges by emphasizing rapid
                    intensity changes.
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900 mb-2">
                    Embossing
                  </h4>
                  <p className="text-sm text-orange-800">
                    Creates 3D effect by emphasizing differences in neighboring
                    pixel values.
                  </p>
                </div>
              </div>

              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded mt-4">
                <p className="font-semibold text-yellow-900 mb-2">
                  Convolution Types:
                </p>
                <ul className="space-y-2 text-yellow-800 text-sm">
                  <li>
                    <strong>Standard:</strong> Processes each pixel by centering
                    kernel over it
                  </li>
                  <li>
                    <strong>Valid (No Padding):</strong> Only where kernel fully
                    overlaps, produces smaller output
                  </li>
                  <li>
                    <strong>Same (Zero Padding):</strong> Adds padding to
                    maintain input dimensions
                  </li>
                  <li>
                    <strong>With Strides:</strong> Skips pixels based on stride
                    value
                  </li>
                </ul>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Edge Detection Techniques">
            <div className="space-y-4 text-slate-700">
              <p>
                Edge detection identifies boundaries within images where rapid
                intensity or color changes occur. These boundaries represent
                important information about image structure.
              </p>

              <Accordion type="single" collapsible className="w-full mt-4">
                <AccordionItem value="canny">
                  <AccordionTrigger>Canny Edge Detector</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Most popular multi-stage edge detection algorithm known
                        for accuracy and noise reduction.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold text-sm mb-2">Steps:</p>
                        <ol className="space-y-1 text-sm ml-4 list-decimal">
                          <li>
                            <strong>Gaussian Smoothing:</strong> Reduce noise
                          </li>
                          <li>
                            <strong>Gradient Calculation:</strong> Compute
                            magnitude and direction
                          </li>
                          <li>
                            <strong>Non-Maximum Suppression:</strong> Thin edges
                          </li>
                          <li>
                            <strong>Hysteresis Thresholding:</strong> Link
                            strong and weak edges using two thresholds
                          </li>
                        </ol>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="log">
                  <AccordionTrigger>
                    Laplacian of Gaussian (LoG)
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Combines Gaussian smoothing with Laplacian edge
                        detection to highlight edges with fine details.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold text-sm mb-2">Steps:</p>
                        <ol className="space-y-1 text-sm ml-4 list-decimal">
                          <li>Apply Gaussian smoothing</li>
                          <li>Apply Laplacian operator (second derivative)</li>
                          <li>
                            Detect zero-crossings (where intensity changes sign)
                          </li>
                          <li>Apply thresholding to obtain binary edge map</li>
                        </ol>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="sobel">
                  <AccordionTrigger>Sobel & Scharr Operators</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Gradient-based methods that approximate image gradients
                        using convolution with specific kernels.
                      </p>
                      <div className="bg-slate-100 p-3 rounded text-sm">
                        <p>
                          <strong>Sobel:</strong> Uses 3×3 kernels for
                          horizontal and vertical gradients
                        </p>
                        <p>
                          <strong>Scharr:</strong> Similar but with better
                          rotational symmetry
                        </p>
                        <p className="mt-2">
                          Both compute gradient magnitude and direction to
                          locate edges
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>

          <ConceptCard title="Corner Detection">
            <div className="space-y-4 text-slate-700">
              <p>
                Corner detection identifies significant points where image
                intensity changes in multiple directions. Corners are invariant
                under translation, rotation, and scale changes, making them
                robust features.
              </p>

              <Accordion type="single" collapsible className="w-full mt-4">
                <AccordionItem value="harris">
                  <AccordionTrigger>Harris Corner Detector</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        One of the earliest and most widely used corner
                        detection methods. Calculates a "cornerness" score based
                        on intensity variation in all directions.
                      </p>
                      <div className="bg-slate-100 p-3 rounded text-sm">
                        <p>
                          <strong>Parameters:</strong>
                        </p>
                        <ul className="ml-4 mt-2 space-y-1">
                          <li>
                            • <strong>block_size:</strong> Neighborhood size for
                            structure tensor
                          </li>
                          <li>
                            • <strong>k:</strong> Empirical constant (0.04-0.06)
                          </li>
                          <li>
                            • <strong>threshold:</strong> Minimum cornerness
                            score to accept
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="shitomasi">
                  <AccordionTrigger>
                    Shi-Tomasi Corner Detector
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Improvement over Harris detector using minimum
                        eigenvalue of gradient matrix for corner selection.
                      </p>
                      <div className="bg-slate-100 p-3 rounded text-sm">
                        <p>
                          <strong>Parameters:</strong>
                        </p>
                        <ul className="ml-4 mt-2 space-y-1">
                          <li>
                            • <strong>maxCorners:</strong> Maximum number of
                            corners to detect
                          </li>
                          <li>
                            • <strong>qualityLevel:</strong> Minimum quality
                            threshold (0-1)
                          </li>
                          <li>
                            • <strong>minDistance:</strong> Minimum distance
                            between corners
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="fast">
                  <AccordionTrigger>
                    FAST (Features from Accelerated Segment Test)
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Machine-learning-based detector designed for real-time
                        applications. Uses decision tree to classify pixels as
                        corners.
                      </p>
                      <div className="bg-slate-100 p-3 rounded text-sm">
                        <p>
                          Examines circle of 16 pixels around candidate point.
                          If sufficient consecutive pixels are brighter or
                          darker than center, it's a corner.
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="orb">
                  <AccordionTrigger>
                    ORB (Oriented FAST and Rotated BRIEF)
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <p className="text-sm">
                        Combines FAST keypoint detector with BRIEF descriptor.
                        Designed for real-time applications with good balance
                        between speed and accuracy.
                      </p>
                      <div className="bg-slate-100 p-3 rounded text-sm">
                        <p className="font-semibold mb-2">Features:</p>
                        <ul className="ml-4 space-y-1">
                          <li>
                            • Rotation invariant (assigns orientation to
                            keypoints)
                          </li>
                          <li>• Scale invariant</li>
                          <li>• Binary descriptor (efficient matching)</li>
                          <li>• Fast computation</li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="HOG Feature Extraction">
            <div className="space-y-4">
              <p className="text-slate-700">
                Extract HOG features to capture gradient information for object
                detection and recognition tasks.
              </p>
              <CodeBlock
                code={`from skimage.feature import hog
from skimage import exposure
import cv2
import matplotlib.pyplot as plt

# Read an image (grayscale)
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute HOG features
# pixels_per_cell: size of each cell (8x8 pixels)
# cells_per_block: number of cells in each block (2x2 cells)
# visualize: return HOG visualization
features, hog_image = hog(image, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)

# Rescale HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('HOG Features')
plt.axis('off')
plt.show()

# Print feature vector shape
print(f"HOG Feature Vector Shape: {features.shape}")
print(f"Number of features: {len(features)}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Local Binary Pattern (LBP)">
            <div className="space-y-4">
              <p className="text-slate-700">
                Extract texture features using LBP for texture classification
                and analysis.
              </p>
              <CodeBlock
                code={`from skimage import feature
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (grayscale)
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute LBP features
radius = 1  # Radius of circular neighborhood
n_points = 8 * radius  # Number of neighboring pixels
lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform')

# Calculate LBP histogram
hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3),
                       range=(0, n_points + 2))

# Normalize histogram
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(lbp_image, cmap='gray')
plt.title('LBP Image')
plt.axis('off')

plt.subplot(122)
plt.bar(range(0, n_points + 2), hist)
plt.title('LBP Histogram')
plt.xlabel('LBP Patterns')
plt.ylabel('Frequency')
plt.show()

print(f"LBP Feature Vector: {hist}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Filtering Operations">
            <div className="space-y-4">
              <p className="text-slate-700">
                Apply various filters for different image processing effects.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# 1. Box Blur - Simple averaging
kernel_box = np.ones((5, 5), dtype=np.float32) / 25
box_blur = cv2.filter2D(image, -1, kernel_box)

# 2. Gaussian Blur - Weighted averaging
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 3. Sobel Edge Detection - Horizontal and vertical gradients
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

# 4. Embossing - 3D effect
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]], dtype=np.float32)
embossed = cv2.filter2D(image, -1, kernel_emboss)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Box Blur', box_blur)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Sobel Edges', sobel_combined.astype(np.uint8))
cv2.imshow('Embossed', embossed)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Convolution Types">
            <div className="space-y-4">
              <p className="text-slate-700">
                Demonstrate different convolution approaches with various border
                handling.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a simple edge detection kernel
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)

# 1. Standard Convolution (default behavior)
standard_conv = cv2.filter2D(image, -1, kernel)

# 2. Valid Convolution (no padding) - smaller output
valid_conv = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# 3. Same Convolution (with padding) - same size output
same_conv = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

# 4. Convolution with stride (downsampling)
# Resize then convolve to simulate stride effect
strided_image = cv2.resize(image, None, fx=0.5, fy=0.5)
strided_conv = cv2.filter2D(strided_image, -1, kernel)

print(f"Original shape: {image.shape}")
print(f"Standard convolution shape: {standard_conv.shape}")
print(f"Valid convolution shape: {valid_conv.shape}")
print(f"Same convolution shape: {same_conv.shape}")
print(f"Strided convolution shape: {strided_conv.shape}")

cv2.imshow('Standard', standard_conv)
cv2.imshow('Valid', valid_conv)
cv2.imshow('Same', same_conv)
cv2.imshow('Strided', strided_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Canny Edge Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                Multi-stage edge detection with noise reduction and edge
                tracking.
              </p>
              <CodeBlock
                code={`import cv2

# Read image in grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise (optional but recommended)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
# threshold1: lower threshold for hysteresis
# threshold2: upper threshold for hysteresis
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Try different threshold values for comparison
edges_low = cv2.Canny(blurred, 30, 100)
edges_high = cv2.Canny(blurred, 100, 200)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Canny Edges (50, 150)', edges)
cv2.imshow('Low Threshold (30, 100)', edges_low)
cv2.imshow('High Threshold (100, 200)', edges_high)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Tip: Ratio of upper:lower threshold should be 2:1 or 3:1`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Laplacian of Gaussian (LoG)">
            <div className="space-y-4">
              <p className="text-slate-700">
                Combine Gaussian smoothing with Laplacian edge detection for
                fine detail preservation.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
smoothed = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Laplacian operator
laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)

# Find zero crossings (convert to absolute values for visualization)
laplacian_abs = np.absolute(laplacian)

# Apply threshold to get binary edge map
_, edges = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
edges = edges.astype(np.uint8)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Smoothed', smoothed)
cv2.imshow('Laplacian', laplacian_abs.astype(np.uint8))
cv2.imshow('LoG Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Harris Corner Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                Detect corners by analyzing intensity variation in all
                directions around each pixel.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Read image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection
# Parameters:
# - gray: input image (must be float32)
# - blockSize: neighborhood size for corner detection
# - ksize: aperture parameter for Sobel derivative
# - k: Harris detector free parameter (0.04-0.06)
corner_scores = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate corner scores to mark corners
corner_scores = cv2.dilate(corner_scores, None)

# Threshold for corner detection (adjust as needed)
threshold = 0.01 * corner_scores.max()

# Mark corners on original image (red dots)
image[corner_scores > threshold] = [0, 0, 255]

cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Shi-Tomasi Corner Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                Improved corner detection using minimum eigenvalue criteria.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Read image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection
# Parameters:
# - maxCorners: maximum number of corners to detect (0 = unlimited)
# - qualityLevel: minimum quality of corner (0-1)
# - minDistance: minimum Euclidean distance between corners
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                   qualityLevel=0.01, 
                                   minDistance=10)

# Convert corners to integer coordinates
corners = np.int0(corners)

# Draw circles at corner locations
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Shi-Tomasi Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Number of corners detected: {len(corners)}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="FAST Corner Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                High-speed corner detection for real-time applications.
              </p>
              <CodeBlock
                code={`import cv2

# Read image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST detector
# Parameters:
# - threshold: intensity difference threshold (default: 40)
# - nonmaxSuppression: apply non-maximum suppression (default: True)
fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)

# Detect keypoints
keypoints = fast.detect(gray, None)

# Draw keypoints on image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                         color=(0, 255, 0))

print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Threshold: {fast.getThreshold()}")
print(f"Non-max suppression: {fast.getNonmaxSuppression()}")

cv2.imshow('FAST Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="ORB Feature Detection and Matching">
            <div className="space-y-4">
              <p className="text-slate-700">
                ORB combines FAST detector with BRIEF descriptor for efficient
                feature matching.
              </p>
              <CodeBlock
                code={`import cv2

# Load two images to match
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Create ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Create Brute Force Matcher
# - NORM_HAMMING: distance metric for binary descriptors
# - crossCheck: only return mutual best matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 20 matches
match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                               matches[:20], None, 
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print(f"Keypoints in image 1: {len(keypoints1)}")
print(f"Keypoints in image 2: {len(keypoints2)}")
print(f"Total matches found: {len(matches)}")
print(f"Top 5 match distances: {[m.distance for m in matches[:5]]}")

cv2.imshow('ORB Matches', match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Histogram of Edge Directions">
            <div className="space-y-4">
              <p className="text-slate-700">
                Extract edge direction distribution for texture and shape
                analysis.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing (optional)
smoothed = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(smoothed, threshold1=30, threshold2=70)

# Calculate gradient direction (orientation)
gradient_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
gradient_orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

# Create histogram of edge directions (8 bins for 360 degrees)
hist, bin_edges = np.histogram(gradient_orientation, bins=8, range=(0, 360))

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge Map')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bin_edges[:-1], hist, width=45, align='center')
plt.title('Histogram of Edge Directions')
plt.xlabel('Edge Direction (degrees)')
plt.ylabel('Frequency')
plt.xticks(range(0, 360, 45))
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"HED Feature Vector: {hist}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Histogram of Intensity Gradients">
            <div className="space-y-4">
              <p className="text-slate-700">
                Capture gradient magnitude and orientation distribution across
                the image.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute gradients using Sobel operator
dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate magnitude and orientation
magnitude = np.sqrt(dx**2 + dy**2)
orientation = np.arctan2(dy, dx) * (180 / np.pi)  # Convert to degrees

# Create histogram of gradient orientations (9 bins for 0-180 degrees)
histogram, bins = np.histogram(orientation, bins=9, range=(0, 180))

# Normalize histogram (optional)
histogram = histogram / histogram.sum()

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(magnitude, cmap='hot')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.bar(bins[:-1], histogram, width=20)
plt.title('Histogram of Intensity Gradients')
plt.xlabel('Gradient Orientation (degrees)')
plt.ylabel('Normalized Frequency')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"HIG Feature Vector: {histogram}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Texture Energy and Contrast">
            <div className="space-y-4">
              <p className="text-slate-700">
                Extract texture features based on energy (uniformity) and
                contrast (variation).
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define neighborhood size (3x3 window)
neighborhood_size = 3

# Calculate texture energy (sum of squared pixel values)
energy = cv2.filter2D(image ** 2, -1, 
                      np.ones((neighborhood_size, neighborhood_size)))

# Calculate texture contrast (standard deviation)
# Mean in neighborhood
mean = cv2.filter2D(image, -1, 
                    np.ones((neighborhood_size, neighborhood_size)) / 
                    (neighborhood_size ** 2))
# Variance in neighborhood
variance = cv2.filter2D(image ** 2, -1, 
                        np.ones((neighborhood_size, neighborhood_size)) / 
                        (neighborhood_size ** 2)) - mean ** 2
contrast = np.sqrt(np.abs(variance))

# Compute histograms
energy_hist, energy_bins = np.histogram(energy, bins=256, range=(0, energy.max()))
contrast_hist, contrast_bins = np.histogram(contrast, bins=256, 
                                             range=(0, contrast.max()))

# Normalize histograms
energy_hist = energy_hist / energy_hist.sum()
contrast_hist = contrast_hist / contrast_hist.sum()

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(energy, cmap='hot')
plt.title('Texture Energy')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(energy_bins[:-1], energy_hist, color='b')
plt.title('Texture Energy Histogram')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(contrast_bins[:-1], contrast_hist, color='r')
plt.title('Texture Contrast Histogram')
plt.xlabel('Contrast')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()`}
              />
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-6 mt-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-green-900 mb-4">
              Lab Tasks
            </h2>
            <p className="text-green-800 mb-4">
              Complete these tasks to practice feature extraction, edge
              detection, and corner detection techniques.
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="task1">
              <AccordionTrigger className="text-lg font-semibold">
                Task 1: Medical Image Analysis for Tumor Detection
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="font-semibold text-blue-900 mb-2">Scenario:</p>
                  <p className="text-blue-800 text-sm">
                    You're part of a medical research team detecting tumors in
                    mammograms. Feature extraction is critical for identifying
                    potential tumor regions.
                  </p>
                </div>

                <div className="space-y-3">
                  <p className="font-semibold text-slate-700">Requirements:</p>
                  <ol className="list-decimal list-inside space-y-2 text-slate-700 text-sm ml-4">
                    <li>
                      Explain how edge detection (Canny or LoG) can identify
                      tumor boundaries in mammograms
                    </li>
                    <li>
                      Describe the step-by-step process of applying edge
                      detection for tumor detection
                    </li>
                    <li>
                      Propose one additional technique (e.g., HOG, LBP, texture
                      analysis) to enhance detection
                    </li>
                    <li>
                      Explain why your chosen technique is suitable for medical
                      imaging
                    </li>
                    <li>Implement both techniques and compare results</li>
                  </ol>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Load mammogram image
mammogram = cv2.imread('mammogram.jpg', cv2.IMREAD_GRAYSCALE)

# Method 1: Canny Edge Detection for tumor boundaries
# Step 1: Denoise with Gaussian blur
denoised = cv2.GaussianBlur(mammogram, (5, 5), 0)

# Step 2: Apply CLAHE for better contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)

# Step 3: Canny edge detection
edges = cv2.Canny(enhanced, 50, 150)

# Step 4: Morphological operations to connect edges
kernel = np.ones((3, 3), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Method 2: LBP for texture analysis
# Tumors often have different texture than healthy tissue
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(enhanced, n_points, radius, method='uniform')

# Create LBP histogram
hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
hist_lbp = hist_lbp.astype("float")
hist_lbp /= (hist_lbp.sum() + 1e-6)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(mammogram, cmap='gray')
axes[0, 0].set_title('Original Mammogram')
axes[0, 0].axis('off')

axes[0, 1].imshow(enhanced, cmap='gray')
axes[0, 1].set_title('Enhanced (CLAHE)')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges_closed, cmap='gray')
axes[0, 2].set_title('Tumor Boundaries (Canny)')
axes[0, 2].axis('off')

axes[1, 0].imshow(lbp, cmap='gray')
axes[1, 0].set_title('LBP Texture Map')
axes[1, 0].axis('off')

axes[1, 1].bar(range(len(hist_lbp)), hist_lbp)
axes[1, 1].set_title('LBP Histogram')
axes[1, 1].set_xlabel('Pattern')
axes[1, 1].set_ylabel('Frequency')

# Combine both methods (edge + texture)
combined = cv2.addWeighted(edges_closed, 0.5, 
                          (lbp * 255 / lbp.max()).astype(np.uint8), 0.5, 0)
axes[1, 2].imshow(combined, cmap='gray')
axes[1, 2].set_title('Combined Detection')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("Analysis Complete")
print(f"Edge pixels detected: {np.sum(edges_closed > 0)}")
print(f"LBP feature vector shape: {hist_lbp.shape}")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task2">
              <AccordionTrigger className="text-lg font-semibold">
                Task 2: Harris Corner Detection
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="font-semibold text-purple-900 mb-2">
                    Objective:
                  </p>
                  <p className="text-purple-800 text-sm">
                    Implement Harris Corner Detection and experiment with
                    different threshold values to understand their effect on
                    corner detection results.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris corner detection with different thresholds
thresholds = [0.001, 0.01, 0.05, 0.1]
results = []

for thresh in thresholds:
    # Create copy of image
    img_copy = image.copy()
    
    # Detect corners
    corner_scores = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corner_scores = cv2.dilate(corner_scores, None)
    
    # Apply threshold
    threshold_value = thresh * corner_scores.max()
    img_copy[corner_scores > threshold_value] = [0, 0, 255]
    
    # Count corners
    num_corners = np.sum(corner_scores > threshold_value)
    results.append((thresh, num_corners, img_copy))
    
    print(f"Threshold: {thresh:.3f} | Corners detected: {num_corners}")

# Display all results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for idx, (thresh, num_corners, img) in enumerate(results):
    row, col = idx // 2, idx % 2
    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'Threshold: {thresh} | Corners: {num_corners}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Analysis
print("\nAnalysis:")
print("- Lower thresholds detect more corners (including weaker ones)")
print("- Higher thresholds detect only strong, prominent corners")
print("- Optimal threshold depends on application requirements")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task3">
              <AccordionTrigger className="text-lg font-semibold">
                Task 3: Real-time Shi-Tomasi Corner Detection from Webcam
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 mb-2">
                    Objective:
                  </p>
                  <p className="text-green-800 text-sm">
                    Access webcam feed and implement Shi-Tomasi corner detection
                    in real-time. Experiment with different parameters.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi
max_corners = 100
quality_level = 0.01
min_distance = 10

print("Press 'q' to quit")
print("Press '+' to increase quality level")
print("Press '-' to decrease quality level")
print("Press 'c' to increase max corners")
print("Press 'v' to decrease max corners")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, 
                                       maxCorners=max_corners,
                                       qualityLevel=quality_level,
                                       minDistance=min_distance)
    
    # Draw corners
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Display parameters
    cv2.putText(frame, f'Max Corners: {max_corners}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Quality: {quality_level:.3f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Detected: {len(corners) if corners is not None else 0}', 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Shi-Tomasi Corner Detection', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        quality_level = min(1.0, quality_level + 0.005)
    elif key == ord('-'):
        quality_level = max(0.001, quality_level - 0.005)
    elif key == ord('c'):
        max_corners += 10
    elif key == ord('v'):
        max_corners = max(10, max_corners - 10)

cap.release()
cv2.destroyAllWindows()

print("\nExperiment Summary:")
print("- Quality level affects corner strength threshold")
print("- Max corners limits the number of detected corners")
print("- Min distance prevents corners from clustering")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task4">
              <AccordionTrigger className="text-lg font-semibold">
                Task 4: Corner Detection for Image Stitching
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-orange-50 p-4 rounded-lg">
                  <p className="font-semibold text-orange-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-orange-800 text-sm">
                    Develop an image stitching application that combines
                    multiple images into a panorama using corner detection for
                    feature matching.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load multiple images for panorama (should have overlap)
images = []
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']

for file in image_files:
    img = cv2.imread(file)
    if img is not None:
        images.append(img)

print(f"Loaded {len(images)} images")

# Function to detect and match corners
def detect_and_match(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect Shi-Tomasi corners
    corners1 = cv2.goodFeaturesToTrack(gray1, maxCorners=500,
                                        qualityLevel=0.01, minDistance=10)
    corners2 = cv2.goodFeaturesToTrack(gray2, maxCorners=500,
                                        qualityLevel=0.01, minDistance=10)
    
    # Use ORB for descriptor computation
    orb = cv2.ORB_create()
    kp1 = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners1]
    kp2 = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=20) for c in corners2]
    
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

# Stitch first two images
if len(images) >= 2:
    img1, img2 = images[0], images[1]
    
    # Detect corners and match
    kp1, kp2, matches = detect_and_match(img1, img2)
    
    print(f"Image 1 corners: {len(kp1)}")
    print(f"Image 2 corners: {len(kp2)}")
    print(f"Matches found: {len(matches)}")
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display corners on individual images
    img1_corners = img1.copy()
    img2_corners = img2.copy()
    for kp in kp1:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img1_corners, (x, y), 3, (0, 255, 0), -1)
    for kp in kp2:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img2_corners, (x, y), 3, (0, 255, 0), -1)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1_corners, cv2.COLOR_BGR2RGB))
    plt.title(f'Image 1 - {len(kp1)} Corners')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2_corners, cv2.COLOR_BGR2RGB))
    plt.title(f'Image 2 - {len(kp2)} Corners')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Feature Matches - Top 50 of {len(matches)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # For actual stitching, use homography
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp and stitch
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        result = cv2.warpPerspective(img1, M, (w1 + w2, h1))
        result[0:h2, 0:w2] = img2
        
        plt.figure(figsize=(15, 5))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Stitched Panorama')
        plt.axis('off')
        plt.show()

print("\nStitching complete using corner-based feature matching!")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task5">
              <AccordionTrigger className="text-lg font-semibold">
                Task 5: ORB Feature Detection and Matching
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-pink-50 p-4 rounded-lg">
                  <p className="font-semibold text-pink-900 mb-2">Objective:</p>
                  <p className="text-pink-800 text-sm">
                    Use ORB detector and descriptor to match features between
                    two images with various transformations (rotation, scaling,
                    perspective).
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images with some common features
image1 = cv2.imread('object1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('object2.jpg', cv2.IMREAD_GRAYSCALE)

# Create ORB detector with parameters
orb = cv2.ORB_create(
    nfeatures=500,        # Maximum number of features to detect
    scaleFactor=1.2,      # Pyramid decimation ratio
    nlevels=8,            # Number of pyramid levels
    edgeThreshold=31,     # Border width where features are not detected
    firstLevel=0,         # Level of pyramid to put source image
    WTA_K=2,              # Number of points for BRIEF descriptor
    scoreType=cv2.ORB_HARRIS_SCORE,  # Harris or FAST score
    patchSize=31          # Size of patch for oriented BRIEF
)

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

print(f"Keypoints in image 1: {len(keypoints1)}")
print(f"Keypoints in image 2: {len(keypoints2)}")
print(f"Descriptor shape: {descriptors1.shape}")

# Method 1: Brute Force Matcher with cross-checking
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_bf = bf.match(descriptors1, descriptors2)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)
# Method 2: BFMatcher with KNN (k=2 for ratio test)
bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING)
matches_knn = bf_knn.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test (Lowe's ratio test)
good_matches = []
for pair in matches_knn:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"\nBrute Force matches: {len(matches_bf)}")
print(f"KNN matches after ratio test: {len(good_matches)}")

# Draw matches - Brute Force
img_bf = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                         matches_bf[:30], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw matches - KNN with ratio test
img_knn = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                          good_matches[:30], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw keypoints
img1_kp = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].imshow(img1_kp, cmap='gray')
axes[0, 0].set_title(f'Image 1 - {len(keypoints1)} Keypoints')
axes[0, 0].axis('off')

axes[0, 1].imshow(img2_kp, cmap='gray')
axes[0, 1].set_title(f'Image 2 - {len(keypoints2)} Keypoints')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_bf, cmap='gray')
axes[1, 0].set_title(f'Brute Force Matches (Top 30 of {len(matches_bf)})')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_knn, cmap='gray')
axes[1, 1].set_title(f'KNN + Ratio Test (Top 30 of {len(good_matches)})')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Analyze match quality
if len(matches_bf) > 0:
    distances_bf = [m.distance for m in matches_bf[:50]]
    print(f"\nBF Match distances (top 50):")
    print(f"  Min: {min(distances_bf):.2f}")
    print(f"  Max: {max(distances_bf):.2f}")
    print(f"  Mean: {np.mean(distances_bf):.2f}")

if len(good_matches) > 0:
    distances_knn = [m.distance for m in good_matches[:50]]
    print(f"\nKNN Match distances (top 50):")
    print(f"  Min: {min(distances_knn):.2f}")
    print(f"  Max: {max(distances_knn):.2f}")
    print(f"  Mean: {np.mean(distances_knn):.2f}")

# Performance Analysis
print("\n" + "="*50)
print("ORB Performance Analysis:")
print("="*50)
print("✓ Rotation Invariant: Yes (orientation assignment)")
print("✓ Scale Invariant: Yes (pyramid levels)")
print("✓ Speed: Very Fast (binary descriptors)")
print("✓ Robustness: Good for most transformations")
print("✓ Best for: Real-time applications, mobile devices")
print("="*50)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task6">
              <AccordionTrigger className="text-lg font-semibold">
                Task 6: Texture Analysis for Material Classification
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-indigo-50 p-4 rounded-lg">
                  <p className="font-semibold text-indigo-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-indigo-800 text-sm">
                    Classify different materials (wood, metal, fabric) using
                    texture feature extraction techniques.
                  </p>
                </div>

                <div className="space-y-3 text-slate-700 text-sm">
                  <p className="font-semibold">Requirements:</p>
                  <ol className="list-decimal list-inside space-y-2 ml-4">
                    <li>
                      Use at least two texture feature extraction techniques
                    </li>
                    <li>Explain how each technique works</li>
                    <li>Discuss advantages and limitations</li>
                    <li>Compare results for different materials</li>
                  </ol>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
import matplotlib.pyplot as plt

# Load images of different materials
materials = {
    'wood': cv2.imread('wood.jpg', cv2.IMREAD_GRAYSCALE),
    'metal': cv2.imread('metal.jpg', cv2.IMREAD_GRAYSCALE),
    'fabric': cv2.imread('fabric.jpg', cv2.IMREAD_GRAYSCALE)
}

print("Material Classification using Texture Features")
print("="*60)

# Technique 1: Local Binary Pattern (LBP)
def extract_lbp_features(image):
    """
    LBP captures local texture patterns by comparing each pixel
    with its neighbors. Good for texture classification.
    
    Advantages:
    - Rotation invariant (with uniform patterns)
    - Computationally efficient
    - Works well with limited data
    
    Limitations:
    - Sensitive to noise
    - Fixed neighborhood size
    """
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                          range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return lbp, hist

# Technique 2: Gray Level Co-occurrence Matrix (GLCM)
def extract_glcm_features(image):
    """
    GLCM captures spatial relationships between pixel intensities.
    Excellent for describing texture properties.
    
    Advantages:
    - Captures spatial relationships
    - Multiple texture properties (contrast, homogeneity, energy)
    - Scale invariant
    
    Limitations:
    - Computationally expensive
    - Requires parameter tuning (distances, angles)
    - Sensitive to rotation
    """
    # Normalize image to 0-255 and convert to uint8
    image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Calculate GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(image_norm, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    
    # Extract texture properties
    contrast = greycoprops(glcm, 'contrast').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    
    features = {
        'contrast': np.mean(contrast),
        'homogeneity': np.mean(homogeneity),
        'energy': np.mean(energy),
        'correlation': np.mean(correlation)
    }
    
    return features

# Extract features for all materials
results = {}
fig, axes = plt.subplots(len(materials), 3, figsize=(15, 12))

for idx, (material_name, image) in enumerate(materials.items()):
    print(f"\n{material_name.upper()}:")
    print("-" * 40)
    
    # LBP Features
    lbp_image, lbp_hist = extract_lbp_features(image)
    print(f"LBP histogram mean: {np.mean(lbp_hist):.4f}")
    print(f"LBP histogram std: {np.std(lbp_hist):.4f}")
    
    # GLCM Features
    glcm_features = extract_glcm_features(image)
    print(f"GLCM Features:")
    for key, value in glcm_features.items():
        print(f"  {key}: {value:.4f}")
    
    # Store results
    results[material_name] = {
        'lbp_hist': lbp_hist,
        'glcm': glcm_features
    }
    
    # Visualize
    axes[idx, 0].imshow(image, cmap='gray')
    axes[idx, 0].set_title(f'{material_name.title()} - Original')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(lbp_image, cmap='gray')
    axes[idx, 1].set_title('LBP Image')
    axes[idx, 1].axis('off')
    
    axes[idx, 2].bar(range(len(lbp_hist)), lbp_hist)
    axes[idx, 2].set_title('LBP Histogram')
    axes[idx, 2].set_xlabel('Pattern')
    axes[idx, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Compare GLCM features across materials
print("\n" + "="*60)
print("GLCM Feature Comparison:")
print("="*60)

glcm_comparison = {}
for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
    glcm_comparison[prop] = {
        mat: results[mat]['glcm'][prop] for mat in materials.keys()
    }

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (prop, values) in enumerate(glcm_comparison.items()):
    materials_list = list(values.keys())
    values_list = list(values.values())
    
    axes[idx].bar(materials_list, values_list, color=['brown', 'gray', 'purple'])
    axes[idx].set_title(f'GLCM {prop.title()}')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis Summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY:")
print("="*60)
print("\nLBP Technique:")
print("  Best for: Quick texture classification, rotation-invariant tasks")
print("  Wood: Often shows periodic patterns (grain)")
print("  Metal: Typically smooth with low variance")
print("  Fabric: Shows repetitive woven patterns")

print("\nGLCM Technique:")
print("  Best for: Detailed texture analysis, medical imaging")
print("  Contrast: High for rough textures (fabric), low for smooth (metal)")
print("  Homogeneity: High for uniform textures (metal), low for varied (wood)")
print("  Energy: High for uniform patterns, low for complex textures")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("Combining LBP and GLCM provides robust material classification")
print("LBP is faster and rotation-invariant")
print("GLCM provides detailed spatial texture information")
print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg">
            <h3 className="font-semibold text-amber-900 mb-2">
              💡 Key Takeaways
            </h3>
            <ul className="space-y-2 text-amber-800">
              <li>
                • Feature extraction reduces data dimensionality while
                preserving important information
              </li>
              <li>
                • Different features are suitable for different applications
                (HOG for objects, LBP for textures)
              </li>
              <li>
                • Edge detection is crucial for boundary identification and
                segmentation
              </li>
              <li>
                • Corner detection provides robust keypoints for matching and
                tracking
              </li>
              <li>
                • ORB is excellent for real-time applications due to its speed
                and efficiency
              </li>
              <li>
                • Combining multiple techniques often yields better results than
                single methods
              </li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
