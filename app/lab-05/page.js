// app/lab-05/page.js
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { BookOpen, Code, CheckCircle, Scissors } from "lucide-react";

export default function Lab05() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">Lab 05: Image Segmentation</h1>
        <p className="text-slate-600 text-lg">
          Thresholding, region-based, edge detection, and clustering-based
          segmentation techniques
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
          <ConceptCard title="What is Image Segmentation?" icon={Scissors}>
            <div className="space-y-4 text-slate-700">
              <p>
                Image segmentation is the process of partitioning an image into
                distinct, non-overlapping regions or segments. Each segment
                corresponds to a meaningful object or part of the scene. The
                goal is to separate areas that share similar visual
                characteristics like color, texture, or intensity.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  Key Objectives:
                </p>
                <ul className="space-y-2 text-blue-800">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Simplify representation:</strong> Make images
                      easier to analyze
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Locate objects:</strong> Identify boundaries of
                      objects and regions
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Extract features:</strong> Separate regions of
                      interest for further processing
                    </span>
                  </li>
                </ul>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mt-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Medical Imaging
                  </h4>
                  <p className="text-sm text-green-800">
                    Identify anatomical structures, tumors in X-rays, CT scans,
                    MRIs for diagnosis and treatment planning.
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Autonomous Vehicles
                  </h4>
                  <p className="text-sm text-purple-800">
                    Detect pedestrians, vehicles, road signs, and lane markings
                    for safe navigation.
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900 mb-2">
                    Satellite Imagery
                  </h4>
                  <p className="text-sm text-orange-800">
                    Classify land cover types, monitor deforestation, assess
                    urban growth and agriculture.
                  </p>
                </div>
                <div className="bg-pink-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-pink-900 mb-2">
                    Augmented Reality
                  </h4>
                  <p className="text-sm text-pink-800">
                    Scene understanding for identifying and tracking objects in
                    real-time video.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Mathematics Behind Segmentation">
            <div className="space-y-4 text-slate-700">
              <p>
                Image segmentation employs various mathematical techniques and
                algorithms:
              </p>

              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="threshold">
                  <AccordionTrigger>Thresholding</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 text-sm">
                      <p>
                        Simple pixel intensity comparison against predefined
                        threshold(s). If intensity is above threshold, assign to
                        one region; otherwise, another.
                      </p>
                      <div className="bg-slate-100 p-3 rounded mt-2">
                        <p className="font-semibold">
                          Formula: f(x,y) = 1 if I(x,y) &gt; T, else 0
                        </p>
                        <p className="text-xs mt-1">
                          Where I(x,y) is pixel intensity and T is threshold
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="edge">
                  <AccordionTrigger>Edge Detection</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 text-sm">
                      <p>
                        Identifies rapid changes in intensity that often
                        correspond to object boundaries. Uses operators like
                        Sobel, Canny, and Laplacian of Gaussian.
                      </p>
                      <div className="bg-slate-100 p-3 rounded mt-2">
                        <p className="font-semibold">
                          Gradient Magnitude: G = √(Gx² + Gy²)
                        </p>
                        <p className="text-xs mt-1">
                          Where Gx and Gy are horizontal and vertical gradients
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="cluster">
                  <AccordionTrigger>Clustering</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 text-sm">
                      <p>
                        Groups similar pixels based on feature similarity
                        (color, texture, intensity). Common methods: K-Means,
                        Mean-Shift, DBSCAN.
                      </p>
                      <div className="bg-slate-100 p-3 rounded mt-2">
                        <p className="font-semibold">
                          K-Means: Minimize Σ||xi - μj||²
                        </p>
                        <p className="text-xs mt-1">
                          Where xi is pixel and μj is cluster centroid
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="graph">
                  <AccordionTrigger>Graph Theory</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-2 text-sm">
                      <p>
                        Pixels as nodes in a graph, edges represent similarity.
                        Algorithms like minimum spanning trees or graph cuts
                        partition the image.
                      </p>
                      <div className="bg-slate-100 p-3 rounded mt-2">
                        <p className="text-xs">
                          Used in advanced techniques like GrabCut and
                          normalized cuts
                        </p>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>

          <ConceptCard title="Segmentation Techniques Overview">
            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    📊 Thresholding
                  </h4>
                  <p className="text-sm text-blue-800 mb-2">
                    Divides based on pixel intensity values
                  </p>
                  <ul className="text-xs text-blue-700 space-y-1">
                    <li>• Global: Single threshold for entire image</li>
                    <li>• Adaptive: Different thresholds for regions</li>
                    <li>• Otsu's: Automatic optimal threshold</li>
                    <li>• Color-based: Multi-channel thresholding</li>
                  </ul>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    🌱 Region Growing
                  </h4>
                  <p className="text-sm text-green-800 mb-2">
                    Starts with seed, expands to similar neighbors
                  </p>
                  <ul className="text-xs text-green-700 space-y-1">
                    <li>• Intensity-based: Similar pixel values</li>
                    <li>• Color-based: Similar color characteristics</li>
                    <li>• Effective for uniform regions</li>
                    <li>• Requires good seed selection</li>
                  </ul>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    💧 Watershed
                  </h4>
                  <p className="text-sm text-purple-800 mb-2">
                    Treats image as topographic map
                  </p>
                  <ul className="text-xs text-purple-700 space-y-1">
                    <li>• Simulates flooding from markers</li>
                    <li>• Great for touching/overlapping objects</li>
                    <li>• Requires marker generation</li>
                    <li>• Uses gradient information</li>
                  </ul>
                </div>

                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900 mb-2">
                    🎯 Clustering
                  </h4>
                  <p className="text-sm text-orange-800 mb-2">
                    Groups pixels by feature similarity
                  </p>
                  <ul className="text-xs text-orange-700 space-y-1">
                    <li>• K-Means: Partitional clustering</li>
                    <li>• Mean-Shift: Density-based</li>
                    <li>• Works with complex characteristics</li>
                    <li>• Requires choosing K (number of clusters)</li>
                  </ul>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Thresholding Techniques">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="global">
                <AccordionTrigger>Global Thresholding</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700 text-sm">
                    <p>
                      Single threshold value applied to entire image. Effective
                      when clear intensity difference exists between objects and
                      background.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">How it works:</p>
                      <ol className="list-decimal list-inside space-y-1 ml-2">
                        <li>
                          Choose threshold value T (e.g., 128 for 0-255 range)
                        </li>
                        <li>
                          For each pixel: if intensity &gt; T, set to max value
                          (255)
                        </li>
                        <li>Otherwise, set to min value (0)</li>
                        <li>Result: Binary image (black and white)</li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="adaptive">
                <AccordionTrigger>Adaptive Thresholding</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700 text-sm">
                    <p>
                      Different threshold values for different regions, adapting
                      to local variations. Ideal for images with varying
                      lighting conditions.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Methods:</p>
                      <ul className="space-y-1 ml-2">
                        <li>
                          • <strong>Mean:</strong> Threshold = mean of
                          neighborhood
                        </li>
                        <li>
                          • <strong>Gaussian:</strong> Weighted mean (Gaussian
                          kernel)
                        </li>
                        <li>
                          • <strong>Parameters:</strong> Block size, constant C
                        </li>
                      </ul>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="otsu">
                <AccordionTrigger>Otsu's Thresholding</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700 text-sm">
                    <p>
                      Automatically selects optimal threshold to maximize
                      inter-class variance. Suitable for bimodal intensity
                      distributions.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">Algorithm:</p>
                      <ol className="list-decimal list-inside space-y-1 ml-2">
                        <li>Compute histogram of image</li>
                        <li>
                          For each possible threshold, calculate within-class
                          variance
                        </li>
                        <li>
                          Select threshold that minimizes within-class variance
                        </li>
                        <li>Equivalently, maximizes between-class variance</li>
                      </ol>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="color">
                <AccordionTrigger>Color-Based Thresholding</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 text-slate-700 text-sm">
                    <p>
                      Applies thresholding in multiple color channels (RGB, HSV)
                      to segment based on color information.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="font-semibold mb-2">
                        HSV Color Space Benefits:
                      </p>
                      <ul className="space-y-1 ml-2">
                        <li>
                          • <strong>Hue:</strong> Color type (0-180° in OpenCV)
                        </li>
                        <li>
                          • <strong>Saturation:</strong> Color purity (0-255)
                        </li>
                        <li>
                          • <strong>Value:</strong> Brightness (0-255)
                        </li>
                        <li>
                          • More intuitive for color segmentation than RGB
                        </li>
                      </ul>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </ConceptCard>

          <ConceptCard title="Region Growing & Watershed">
            <div className="space-y-4 text-slate-700">
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">
                  Region Growing
                </h4>
                <p className="text-sm text-green-800 mb-3">
                  Groups neighboring pixels with similar properties into
                  segments. Starts with seed pixels and expands to adjacent
                  similar pixels.
                </p>
                <div className="text-xs text-green-700 space-y-1">
                  <p>
                    <strong>Steps:</strong>
                  </p>
                  <ol className="list-decimal list-inside ml-2 space-y-1">
                    <li>Select seed pixel(s) in region of interest</li>
                    <li>
                      Define similarity criterion (e.g., intensity difference
                      &lt; threshold)
                    </li>
                    <li>Check neighboring pixels; if similar, add to region</li>
                    <li>Repeat until no more pixels meet criterion</li>
                    <li>
                      Result: Segmented region with uniform characteristics
                    </li>
                  </ol>
                </div>
              </div>

              <div className="bg-blue-50 p-4 rounded-lg mt-4">
                <h4 className="font-semibold text-blue-900 mb-2">
                  Watershed Segmentation
                </h4>
                <p className="text-sm text-blue-800 mb-3">
                  Treats gradient image as topographic surface. Simulates
                  flooding from markers to identify segment boundaries.
                  Excellent for touching/overlapping objects.
                </p>
                <div className="text-xs text-blue-700 space-y-2">
                  <p>
                    <strong>Steps:</strong>
                  </p>
                  <ol className="list-decimal list-inside ml-2 space-y-1">
                    <li>
                      <strong>Preprocessing:</strong> Convert to grayscale,
                      enhance contrast
                    </li>
                    <li>
                      <strong>Marker Generation:</strong> Create markers
                      (manual, threshold, distance transform)
                    </li>
                    <li>
                      <strong>Gradient Calculation:</strong> Compute gradient
                      (Sobel/Scharr)
                    </li>
                    <li>
                      <strong>Marker Labeling:</strong> Label markers with
                      different integers
                    </li>
                    <li>
                      <strong>Watershed Transform:</strong> Fill basins from
                      markers
                    </li>
                    <li>
                      <strong>Post-processing:</strong> Apply morphological
                      operations to refine
                    </li>
                  </ol>
                  <p className="mt-2">
                    <strong>Key Concept:</strong> High gradients = edges =
                    watershed boundaries
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Clustering-Based Segmentation">
            <div className="space-y-4 text-slate-700">
              <p>
                Groups pixels into clusters based on similarity criteria like
                color or intensity. K-Means is the most popular clustering
                algorithm for image segmentation.
              </p>

              <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                <p className="font-semibold text-purple-900 mb-2">
                  K-Means Algorithm Steps:
                </p>
                <ol className="list-decimal list-inside space-y-2 text-purple-800 text-sm ml-2">
                  <li>
                    <strong>Feature Extraction:</strong> Choose features (RGB
                    values, LAB, HSV, intensity)
                  </li>
                  <li>
                    <strong>Initialize K Centroids:</strong> Select number of
                    clusters and initial centers
                  </li>
                  <li>
                    <strong>Assignment Step:</strong> Assign each pixel to
                    nearest centroid (Euclidean distance)
                  </li>
                  <li>
                    <strong>Update Step:</strong> Recalculate centroids as mean
                    of assigned pixels
                  </li>
                  <li>
                    <strong>Convergence:</strong> Repeat until assignments don't
                    change significantly
                  </li>
                  <li>
                    <strong>Segmentation:</strong> Create segmented image based
                    on final cluster assignments
                  </li>
                </ol>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mt-4">
                <div className="bg-green-50 p-3 rounded">
                  <h5 className="font-semibold text-green-900 text-sm mb-2">
                    ✓ Advantages
                  </h5>
                  <ul className="text-xs text-green-700 space-y-1">
                    <li>• Simple and fast algorithm</li>
                    <li>• Works well for distinct clusters</li>
                    <li>• Scalable to large images</li>
                    <li>• Easy to implement</li>
                  </ul>
                </div>
                <div className="bg-red-50 p-3 rounded">
                  <h5 className="font-semibold text-red-900 text-sm mb-2">
                    ✗ Limitations
                  </h5>
                  <ul className="text-xs text-red-700 space-y-1">
                    <li>• Must specify K beforehand</li>
                    <li>• Sensitive to initialization</li>
                    <li>• Assumes spherical clusters</li>
                    <li>• Can converge to local optima</li>
                  </ul>
                </div>
              </div>

              <div className="bg-slate-100 p-4 rounded mt-4">
                <p className="text-sm font-semibold mb-2">
                  Choosing K (Number of Clusters):
                </p>
                <ul className="text-xs space-y-1 ml-2">
                  <li>
                    • <strong>Elbow Method:</strong> Plot within-cluster sum of
                    squares vs K
                  </li>
                  <li>
                    • <strong>Silhouette Analysis:</strong> Measure cluster
                    cohesion and separation
                  </li>
                  <li>
                    • <strong>Domain Knowledge:</strong> Use prior knowledge
                    about image content
                  </li>
                  <li>
                    • <strong>Trial and Error:</strong> Test different K values
                    and visually inspect
                  </li>
                </ul>
              </div>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="Global Thresholding">
            <div className="space-y-4">
              <p className="text-slate-700">
                Simple binary thresholding with a single global threshold value.
              </p>
              <CodeBlock
                code={`import cv2
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply global thresholding
# Syntax: cv2.threshold(src, thresh, maxval, type)
# thresh: threshold value (128 for mid-range)
# maxval: maximum value to use (255 for white)
# type: thresholding type
ret, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Try different thresholds
_, thresh_100 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
_, thresh_150 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
_, thresh_inv = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(binary_mask, cmap='gray')
axes[0, 1].set_title('Threshold = 128')
axes[0, 1].axis('off')

axes[0, 2].imshow(thresh_100, cmap='gray')
axes[0, 2].set_title('Threshold = 100')
axes[0, 2].axis('off')

axes[1, 0].imshow(thresh_150, cmap='gray')
axes[1, 0].set_title('Threshold = 150')
axes[1, 0].axis('off')

axes[1, 1].imshow(thresh_inv, cmap='gray')
axes[1, 1].set_title('Inverted (128)')
axes[1, 1].axis('off')

# Histogram
axes[1, 2].hist(image.ravel(), bins=256, range=(0, 256))
axes[1, 2].axvline(x=128, color='r', linestyle='--', label='Threshold')
axes[1, 2].set_title('Histogram')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

print(f"Threshold value used: {ret}")
print(f"Pixels above threshold: {np.sum(binary_mask == 255)}")
print(f"Pixels below threshold: {np.sum(binary_mask == 0)}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Adaptive Thresholding">
            <div className="space-y-4">
              <p className="text-slate-700">
                Different threshold values for different regions, ideal for
                varying lighting conditions.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Global thresholding for comparison
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Mean Thresholding
# Threshold = mean of neighborhood area minus constant C
adaptive_mean = cv2.adaptiveThreshold(
    image, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C,  # Use mean of neighborhood
    cv2.THRESH_BINARY, 
    11,  # Block size (neighborhood area)
    2    # Constant C (subtracted from mean)
)

# Adaptive Gaussian Thresholding
# Threshold = weighted mean (Gaussian window) minus constant C
adaptive_gaussian = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted mean
    cv2.THRESH_BINARY,
    11,  # Block size
    2    # Constant C
)

# Try different block sizes
adaptive_small = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 7, 2  # Smaller block
)

adaptive_large = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 21, 2  # Larger block
)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(global_thresh, cmap='gray')
axes[0, 1].set_title('Global Thresholding')
axes[0, 1].axis('off')

axes[0, 2].imshow(adaptive_mean, cmap='gray')
axes[0, 2].set_title('Adaptive Mean')
axes[0, 2].axis('off')

axes[1, 0].imshow(adaptive_gaussian, cmap='gray')
axes[1, 0].set_title('Adaptive Gaussian (Block=11)')
axes[1, 0].axis('off')

axes[1, 1].imshow(adaptive_small, cmap='gray')
axes[1, 1].set_title('Adaptive Gaussian (Block=7)')
axes[1, 1].axis('off')

axes[1, 2].imshow(adaptive_large, cmap='gray')
axes[1, 2].set_title('Adaptive Gaussian (Block=21)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("Adaptive thresholding works well for:")
print("- Documents with uneven lighting")
print("- Images with shadows")
print("- Varying illumination across image")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Otsu's Thresholding">
            <div className="space-y-4">
              <p className="text-slate-700">
                Automatically determines optimal threshold value for bimodal
                distributions.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
# Returns optimal threshold value and binary image
ret_otsu, otsu_thresh = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Compare with manual threshold
_, manual_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply Gaussian blur before Otsu (often improves results)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
ret_otsu_blur, otsu_blur = cv2.threshold(
    blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(manual_thresh, cmap='gray')
axes[0, 1].set_title('Manual Threshold (127)')
axes[0, 1].axis('off')

axes[0, 2].imshow(otsu_thresh, cmap='gray')
axes[0, 2].set_title(f"Otsu's Method (T={ret_otsu:.0f})")
axes[0, 2].axis('off')

axes[1, 0].imshow(blurred, cmap='gray')
axes[1, 0].set_title('Blurred Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(otsu_blur, cmap='gray')
axes[1, 1].set_title(f"Otsu on Blurred (T={ret_otsu_blur:.0f})")
axes[1, 1].axis('off')

# Plot histogram with threshold line
axes[1, 2].hist(image.ravel(), bins=256, range=(0, 256), alpha=0.7)
axes[1, 2].axvline(x=ret_otsu, color='r', linestyle='--', 
                   linewidth=2, label=f"Otsu's T={ret_otsu:.0f}")
axes[1, 2].axvline(x=127, color='b', linestyle='--', 
                   linewidth=2, label='Manual T=127')
axes[1, 2].set_title('Histogram with Thresholds')
axes[1, 2].set_xlabel('Pixel Intensity')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Otsu's optimal threshold: {ret_otsu:.2f}")
print(f"Otsu on blurred optimal threshold: {ret_otsu_blur:.2f}")
print("\nOtsu's method automatically finds the threshold that")
print("best separates the two peaks in a bimodal histogram.")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Color-Based Thresholding (HSV)">
            <div className="space-y-4">
              <p className="text-slate-700">
                Segment objects based on color using HSV color space for better
                color isolation.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load color image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges in HSV
# Example: Detecting red objects
# Red hue is at 0° and 180° in OpenCV (wraps around)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

# Detecting green objects
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Detecting blue objects
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Apply masks to original image
result_red = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_red)
result_green = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_green)
result_blue = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_blue)

# Display results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(hsv_image)
axes[0, 1].set_title('HSV Image')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_red, cmap='gray')
axes[0, 2].set_title('Red Mask')
axes[0, 2].axis('off')

axes[0, 3].imshow(result_red)
axes[0, 3].set_title('Red Objects')
axes[0, 3].axis('off')

axes[1, 0].imshow(mask_green, cmap='gray')
axes[1, 0].set_title('Green Mask')
axes[1, 0].axis('off')

axes[1, 1].imshow(result_green)
axes[1, 1].set_title('Green Objects')
axes[1, 1].axis('off')

axes[1, 2].imshow(mask_blue, cmap='gray')
axes[1, 2].set_title('Blue Mask')
axes[1, 2].axis('off')

axes[1, 3].imshow(result_blue)
axes[1, 3].set_title('Blue Objects')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

# HSV ranges guide
print("HSV Color Ranges (OpenCV):")
print("-" * 40)
print("Red:    [0-10, 170-180] (wraps around)")
print("Orange: [10-25]")
print("Yellow: [25-35]")
print("Green:  [35-85]")
print("Cyan:   [85-100]")
print("Blue:   [100-130]")
print("Purple: [130-160]")
print("Pink:   [160-170]")
print("\nS (Saturation): 50-255 (avoid low saturation)")
print("V (Value): 50-255 (avoid too dark)")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Watershed Segmentation">
            <div className="space-y-4">
              <p className="text-slate-700">
                Separate touching or overlapping objects using watershed
                algorithm with marker-based approach.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('coins.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Preprocessing - noise reduction
denoised = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Thresholding to get binary image
_, binary = cv2.threshold(denoised, 0, 255, 
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Sure background area (dilate)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 5: Sure foreground area (distance transform)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 
                            255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)

# Step 6: Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 7: Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Add 1 so background is not 0
markers[unknown == 255] = 0  # Mark unknown region as 0

# Step 8: Apply watershed
markers = cv2.watershed(image, markers)
image_rgb[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# Display results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('Binary (Otsu)')
axes[0, 1].axis('off')

axes[0, 2].imshow(opening, cmap='gray')
axes[0, 2].set_title('Opening (Noise Removal)')
axes[0, 2].axis('off')

axes[0, 3].imshow(sure_bg, cmap='gray')
axes[0, 3].set_title('Sure Background')
axes[0, 3].axis('off')

axes[1, 0].imshow(dist_transform, cmap='hot')
axes[1, 0].set_title('Distance Transform')
axes[1, 0].axis('off')

axes[1, 1].imshow(sure_fg, cmap='gray')
axes[1, 1].set_title('Sure Foreground')
axes[1, 1].axis('off')

axes[1, 2].imshow(unknown, cmap='gray')
axes[1, 2].set_title('Unknown Region')
axes[1, 2].axis('off')

axes[1, 3].imshow(image_rgb)
axes[1, 3].set_title('Watershed Result')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

# Count objects
num_objects = len(np.unique(markers)) - 2  # Subtract background and boundary
print(f"Number of objects detected: {num_objects}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="K-Means Clustering Segmentation">
            <div className="space-y-4">
              <p className="text-slate-700">
                Segment images by grouping similar pixels using K-Means
                clustering algorithm.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image to 2D array of pixels (rows x 3 color values)
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define stopping criteria for K-Means
# (type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Try different numbers of clusters
k_values = [2, 3, 5, 8]
results = []

for k in k_values:
    print(f"Applying K-Means with K={k}...")
    
    # Apply K-Means
    # Returns: compactness, labels, centers
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, 
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Map labels to center values
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to original image shape
    segmented_image = segmented_image.reshape(image_rgb.shape)
    
    results.append((k, segmented_image))

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

for idx, (k, seg_img) in enumerate(results):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(seg_img)
    axes[row, col].set_title(f'K-Means (K={k})')
    axes[row, col].axis('off')

# Hide last subplot if odd number
if len(results) < 5:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Detailed example with K=3
print("\nDetailed K-Means with K=3:")
k = 3
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10,
                                 cv2.KMEANS_RANDOM_CENTERS)

# Show cluster colors
print(f"\nCluster Centers (RGB):")
for i, center in enumerate(centers):
    print(f"  Cluster {i}: RGB{tuple(center.astype(int))}")

# Count pixels in each cluster
unique, counts = np.unique(labels, return_counts=True)
print(f"\nPixels per cluster:")
for cluster, count in zip(unique, counts):
    percentage = (count / len(labels)) * 100
    print(f"  Cluster {cluster}: {count} pixels ({percentage:.1f}%)")

print("\nK-Means Tips:")
print("- Lower K: Fewer, broader regions")
print("- Higher K: More detailed segmentation")
print("- Choose K based on application needs")
print("- Try multiple K values and compare results")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Region Growing Segmentation">
            <div className="space-y-4">
              <p className="text-slate-700">
                Grow regions from seed points based on intensity similarity.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed, threshold=10):
    """
    Region growing segmentation.
    
    Parameters:
    - image: grayscale image
    - seed: (x, y) seed point
    - threshold: intensity difference threshold
    """
    h, w = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros((h, w), dtype=bool)
    
    # 8-connectivity neighbors
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    # Queue for BFS
    queue = [seed]
    visited[seed[1], seed[0]] = True
    seed_value = float(image[seed[1], seed[0]])
    
    while queue:
        x, y = queue.pop(0)
        segmented[y, x] = 255
        
        # Check all neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < w and 0 <= ny < h:
                # Check if not visited and similar intensity
                if not visited[ny, nx]:
                    if abs(float(image[ny, nx]) - seed_value) < threshold:
                        queue.append((nx, ny))
                        visited[ny, nx] = True
    
    return segmented

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define multiple seed points
seeds = [(100, 150), (200, 200), (300, 100)]  # (x, y) coordinates
thresholds = [10, 20, 30]

# Apply region growing with different parameters
results = []
for seed in seeds:
    for thresh in [10, 20]:
        result = region_growing(image, seed, threshold=thresh)
        results.append((seed, thresh, result))

# Display results
num_results = len(results)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
for seed in seeds:
    axes[0, 0].plot(seed[0], seed[1], 'r*', markersize=15)
axes[0, 0].axis('off')

for idx, (seed, thresh, result) in enumerate(results[:7]):
    row = (idx + 1) // 4
    col = (idx + 1) % 4
    axes[row, col].imshow(result, cmap='gray')
    axes[row, col].set_title(f'Seed {seed}, T={thresh}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print("Region Growing Parameters:")
print("- Seed point: Starting pixel for growth")
print("- Threshold: Max intensity difference allowed")
print("- Lower threshold: Stricter, smaller regions")
print("- Higher threshold: More lenient, larger regions")`}
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
              Complete these tasks to master image segmentation techniques.
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="task1">
              <AccordionTrigger className="text-lg font-semibold">
                Task 1: Thresholding-Based Segmentation (Bone Fracture)
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="font-semibold text-blue-900 mb-2">Scenario:</p>
                  <p className="text-blue-800 text-sm">
                    You have a grayscale medical X-ray image of a bone fracture.
                    The fracture area is significantly darker than the
                    surrounding bone. Use thresholding to isolate the fracture.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load X-ray image
xray = cv2.imread('bone_xray.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization for better contrast
equalized = cv2.equalizeHist(xray)

# Try different thresholding methods
# 1. Global thresholding
_, global_thresh = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY_INV)

# 2. Otsu's automatic thresholding
_, otsu_thresh = cv2.threshold(equalized, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(equalized, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to clean up
kernel = np.ones((3, 3), np.uint8)
global_clean = cv2.morphologyEx(global_thresh, cv2.MORPH_CLOSE, kernel)
global_clean = cv2.morphologyEx(global_clean, cv2.MORPH_OPEN, kernel)

otsu_clean = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
otsu_clean = cv2.morphologyEx(otsu_clean, cv2.MORPH_OPEN, kernel)

# Highlight fracture on original image
fracture_highlight = cv2.cvtColor(xray, cv2.COLOR_GRAY2BGR)
fracture_highlight[otsu_clean == 255] = [0, 0, 255]  # Red for fracture

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(xray, cmap='gray')
axes[0, 0].set_title('Original X-ray')
axes[0, 0].axis('off')

axes[0, 1].imshow(equalized, cmap='gray')
axes[0, 1].set_title('Histogram Equalized')
axes[0, 1].axis('off')

axes[0, 2].imshow(global_thresh, cmap='gray')
axes[0, 2].set_title('Global Threshold (T=100)')
axes[0, 2].axis('off')

axes[1, 0].imshow(otsu_thresh, cmap='gray')
axes[1, 0].set_title("Otsu's Threshold")
axes[1, 0].axis('off')

axes[1, 1].imshow(adaptive_thresh, cmap='gray')
axes[1, 1].set_title('Adaptive Threshold')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(fracture_highlight, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Fracture Highlighted')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Calculate fracture area
fracture_pixels = np.sum(otsu_clean == 255)
total_pixels = otsu_clean.size
fracture_percentage = (fracture_pixels / total_pixels) * 100

print(f"Fracture Analysis:")
print(f"  Total pixels: {total_pixels}")
print(f"  Fracture pixels: {fracture_pixels}")
print(f"  Fracture area: {fracture_percentage:.2f}%")
print(f"\nBest method: Otsu's thresholding (automatic)")
print(f"Reason: Automatically finds optimal threshold for bimodal distribution")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task2">
              <AccordionTrigger className="text-lg font-semibold">
                Task 2: Region Growing for Cell Segmentation
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="font-semibold text-purple-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-purple-800 text-sm">
                    You have a microscopic image of cells. Choose a seed point
                    in one of the cells and perform region growing to identify
                    and separate that cell from the rest.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing_advanced(image, seed, threshold=15):
    """Enhanced region growing with statistics"""
    h, w = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros((h, w), dtype=bool)
    
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    queue = [seed]
    visited[seed[1], seed[0]] = True
    seed_value = float(image[seed[1], seed[0]])
    
    # Keep track of region statistics
    pixel_count = 0
    sum_intensity = 0
    
    while queue:
        x, y = queue.pop(0)
        segmented[y, x] = 255
        pixel_count += 1
        sum_intensity += image[y, x]
        
        # Calculate running mean for adaptive threshold
        mean_intensity = sum_intensity / pixel_count
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                # Use adaptive threshold based on mean
                if abs(float(image[ny, nx]) - mean_intensity) < threshold:
                    queue.append((nx, ny))
                    visited[ny, nx] = True
    
    return segmented, pixel_count

# Load microscopic image
cells_image = cv2.imread('cells.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocess: enhance contrast and reduce noise
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(cells_image)
denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Interactive seed selection (for demo, we'll use predefined seeds)
seeds = [
    (150, 200, 'Cell 1'),
    (300, 150, 'Cell 2'),
    (250, 350, 'Cell 3')
]

# Apply region growing for each seed
results = []
for seed_x, seed_y, label in seeds:
    for threshold in [10, 20, 30]:
        segmented, pixel_count = region_growing_advanced(
            denoised, (seed_x, seed_y), threshold
        )
        results.append((label, threshold, segmented, pixel_count))

# Display results
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Original with seed points
axes[0, 0].imshow(cells_image, cmap='gray')
axes[0, 0].set_title('Original with Seeds')
for seed_x, seed_y, label in seeds:
    axes[0, 0].plot(seed_x, seed_y, 'r*', markersize=15)
    axes[0, 0].text(seed_x+10, seed_y-10, label, color='red', fontsize=8)
axes[0, 0].axis('off')

axes[0, 1].imshow(enhanced, cmap='gray')
axes[0, 1].set_title('CLAHE Enhanced')
axes[0, 1].axis('off')

axes[0, 2].imshow(denoised, cmap='gray')
axes[0, 2].set_title('Denoised')
axes[0, 2].axis('off')

# Combined result
combined = np.zeros_like(cells_image)
colors = [85, 170, 255]  # Different gray levels for different cells
for i, (label, threshold, seg, _) in enumerate(results[::3]):  # Every 3rd (first threshold)
    combined[seg == 255] = colors[i]

axes[0, 3].imshow(combined, cmap='gray')
axes[0, 3].set_title('All Cells Segmented')
axes[0, 3].axis('off')

# Individual cell results
idx = 0
for i in range(3):  # 3 cells
    for j in range(3):  # 3 thresholds
        if idx < len(results):
            label, threshold, segmented, pixel_count = results[idx]
            row = i + 1
            col = j + 1 if j < 3 else 0
            
            axes[row, col].imshow(segmented, cmap='gray')
            axes[row, col].set_title(f'{label}, T={threshold}\n{pixel_count} pixels')
            axes[row, col].axis('off')
            idx += 1

plt.tight_layout()
plt.show()

# Analyze results
print("Cell Segmentation Analysis:")
print("=" * 60)
for label, threshold, segmented, pixel_count in results[::3]:
    print(f"\n{label}:")
    print(f"  Threshold: {threshold}")
    print(f"  Pixels: {pixel_count}")
    print(f"  Approximate area: {pixel_count * 0.01:.2f} units²")

print("\n" + "=" * 60)
print("Observations:")
print("- Lower threshold: More conservative, captures only very similar pixels")
print("- Higher threshold: More aggressive, may include background")
print("- Optimal threshold depends on cell characteristics")
print("- Preprocessing (CLAHE, denoising) improves results")`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task3">
              <AccordionTrigger className="text-lg font-semibold">
                Task 3: Watershed Segmentation for Coin Counting
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 mb-2">Scenario:</p>
                  <p className="text-green-800 text-sm">
                    You have an image of overlapping coins on a table. Use
                    watershed segmentation to separate and count the individual
                    coins.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# Load image
coins = cv2.imread('coins.jpg')
coins_rgb = cv2.cvtColor(coins, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

# Step 1: Preprocessing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Thresholding
_, binary = cv2.threshold(blurred, 0, 255, 
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Noise removal with morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Sure background (dilation)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 5: Finding sure foreground using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Try different threshold percentages for distance transform
thresholds = [0.3, 0.5, 0.7]
results_watershed = []

for thresh_pct in thresholds:
    # Sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 
                                thresh_pct * dist_transform.max(), 
                                255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers_copy = markers.copy()
    markers_watershed = cv2.watershed(coins, markers_copy)
    
    # Create visualization
    result_img = coins_rgb.copy()
    result_img[markers_watershed == -1] = [255, 0, 0]  # Boundaries in red
    
    # Count coins (exclude background and boundaries)
    num_coins = len(np.unique(markers_watershed)) - 2
    
    results_watershed.append({
        'threshold': thresh_pct,
        'sure_fg': sure_fg,
        'markers': markers,
        'result': result_img,
        'num_coins': num_coins
    })

# Create colored visualization for best result
best_result = results_watershed[1]  # Middle threshold usually best
markers_colored = best_result['markers'].copy()

# Generate random colors for each coin
np.random.seed(42)
colors = np.random.randint(0, 255, (markers_colored.max() + 1, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # Background black
colored_markers = colors[markers_colored]

# Display comprehensive results
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Preprocessing steps
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(coins_rgb)
ax1.set_title('Original Image')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(binary, cmap='gray')
ax2.set_title('Binary (Otsu)')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(opening, cmap='gray')
ax3.set_title('Opening (Noise Removal)')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(dist_transform, cmap='hot')
ax4.set_title('Distance Transform')
ax4.axis('off')

# Row 2: Different threshold results
for idx, result in enumerate(results_watershed):
    ax = fig.add_subplot(gs[1, idx])
    ax.imshow(result['sure_fg'], cmap='gray')
    ax.set_title(f"Sure FG (T={result['threshold']})")
    ax.axis('off')

# Empty spot
ax = fig.add_subplot(gs[1, 3])
ax.imshow(sure_bg, cmap='gray')
ax.set_title('Sure Background')
ax.axis('off')

# Row 3: Watershed results
for idx, result in enumerate(results_watershed):
    ax = fig.add_subplot(gs[2, idx])
    ax.imshow(result['result'])
    ax.set_title(f"Watershed T={result['threshold']}\n{result['num_coins']} coins")
    ax.axis('off')

# Colored markers
ax = fig.add_subplot(gs[2, 3])
ax.imshow(colored_markers)
ax.set_title('Colored Regions')
ax.axis('off')

plt.suptitle('Watershed Segmentation for Coin Counting', fontsize=16, y=0.98)
plt.show()

# Detailed analysis
print("=" * 60)
print("COIN COUNTING ANALYSIS")
print("=" * 60)

for result in results_watershed:
    print(f"\nThreshold = {result['threshold']}:")
    print(f"  Coins detected: {result['num_coins']}")

best = results_watershed[1]
print(f"\n{'='*60}")
print(f"BEST RESULT: Threshold = {best['threshold']}")
print(f"Total coins detected: {best['num_coins']}")
print(f"{'='*60}")

# Calculate individual coin areas
markers_best = best['markers']
for coin_id in range(1, best['num_coins'] + 1):
    coin_area = np.sum(markers_best == coin_id)
    print(f"Coin {coin_id}: {coin_area} pixels")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Distance transform threshold affects coin separation")
print("2. Lower threshold (0.3): May over-segment")
print("3. Medium threshold (0.5): Usually optimal")
print("4. Higher threshold (0.7): May merge touching coins")
print("5. Morphological operations critical for noise removal")
print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task4">
              <AccordionTrigger className="text-lg font-semibold">
                Task 4: K-Means Clustering for Flower Segmentation
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-pink-50 p-4 rounded-lg">
                  <p className="font-semibold text-pink-900 mb-2">Scenario:</p>
                  <p className="text-pink-800 text-sm">
                    You have an image of colorful flowers in a garden. Use
                    K-Means clustering to separate different types of flowers
                    based on color.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
flowers = cv2.imread('flowers.jpg')
flowers_rgb = cv2.cvtColor(flowers, cv2.COLOR_BGR2RGB)
original_shape = flowers_rgb.shape

# Prepare data for clustering
pixel_values = flowers_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define K-Means parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Try different numbers of clusters
k_values = [2, 3, 4, 5, 6, 8]
results = []

print("Applying K-Means clustering...")
for k in k_values:
    print(f"  K = {k}...", end=' ')
    
    # OpenCV K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10,
                                     cv2.KMEANS_PP_CENTERS)
    
    # Convert to uint8
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(original_shape)
    
    # Create individual cluster masks
    labels_2d = labels.reshape((original_shape[0], original_shape[1]))
    cluster_masks = []
    for i in range(k):
        mask = np.uint8((labels_2d == i) * 255)
        cluster_masks.append(mask)
    
    results.append({
        'k': k,
        'segmented': segmented_image,
        'labels': labels_2d,
        'centers': centers,
        'masks': cluster_masks
    })
    print("Done")

# Display main results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(flowers_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

for idx, result in enumerate(results):
    if idx < 7:
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        axes[row, col].imshow(result['segmented'])
        axes[row, col].set_title(f"K-Means (K={result['k']})")
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Detailed analysis for K=5 (good balance)
k_optimal = 5
optimal_result = [r for r in results if r['k'] == k_optimal][0]

# Show individual clusters
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(flowers_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(optimal_result['segmented'])
axes[0, 1].set_title(f'Segmented (K={k_optimal})')
axes[0, 1].axis('off')

# Show dominant colors
dominant_colors = np.zeros((100, 500, 3), dtype=np.uint8)
x_offset = 0
for i, center in enumerate(optimal_result['centers']):
    dominant_colors[:, x_offset:x_offset+100] = center
    x_offset += 100

axes[0, 2].imshow(dominant_colors)
axes[0, 2].set_title('Dominant Colors')
axes[0, 2].axis('off')

# Show individual cluster masks
for i in range(min(5, k_optimal)):
    row = 1
    col = i if i < 3 else i - 3
    if col < 3:
        mask = optimal_result['masks'][i]
        masked_image = flowers_rgb.copy()
        masked_image[mask == 0] = [255, 255, 255]  # White background
        
        axes[row, col].imshow(masked_image)
        axes[row, col].set_title(f'Cluster {i+1}')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Statistical analysis
print("\n" + "="*60)
print(f"DETAILED ANALYSIS FOR K={k_optimal}")
print("="*60)

labels_optimal = optimal_result['labels']
total_pixels = labels_optimal.size

print(f"\nCluster Centers (RGB Colors):")
for i, center in enumerate(optimal_result['centers']):
    print(f"  Cluster {i+1}: RGB{tuple(center)}")

print(f"\nCluster Distribution:")
for i in range(k_optimal):
    count = np.sum(labels_optimal == i)
    percentage = (count / total_pixels) * 100
    print(f"  Cluster {i+1}: {count:,} pixels ({percentage:.2f}%)")

# Color-based flower type identification
print(f"\n{'='*60}")
print("FLOWER TYPE IDENTIFICATION (Based on Color)")
print("="*60)

def identify_flower_type(rgb):
    """Simple heuristic to identify flower type by color"""
    r, g, b = rgb
    if r > 180 and g < 100 and b < 100:
        return "Red flowers (e.g., roses)"
    elif r > 200 and g > 150 and b < 100:
        return "Yellow flowers (e.g., sunflowers)"
    elif r > 180 and g < 150 and b > 150:
        return "Pink flowers (e.g., cherry blossoms)"
    elif r < 100 and g < 100 and b > 150:
        return "Blue/Purple flowers (e.g., lavender)"
    elif r < 100 and g > 150 and b < 100:
        return "Green (leaves/stems)"
    else:
        return "Mixed/Other"

for i, center in enumerate(optimal_result['centers']):
    flower_type = identify_flower_type(center)
    print(f"Cluster {i+1}: {flower_type}")

print(f"\n{'='*60}")
print("RECOMMENDATIONS:")
print("="*60)
print("- K=2-3: Too few clusters, poor separation")
print("- K=4-6: Good balance, distinct flower types")
print("- K=7+: Over-segmentation, may split single flowers")
print("- Optimal K depends on image complexity")
print("- Consider using elbow method or silhouette analysis")
print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg">
            <h3 className="font-semibold text-amber-900 mb-2">
              💡 Segmentation Best Practices
            </h3>
            <ul className="space-y-2 text-amber-800 text-sm">
              <li>
                • <strong>Preprocessing is crucial:</strong> Always denoise and
                enhance contrast before segmentation
              </li>
              <li>
                • <strong>Choose method based on image:</strong> Thresholding
                for simple images, clustering for complex ones
              </li>
              <li>
                • <strong>Watershed for touching objects:</strong> Best for
                separating overlapping or connected regions
              </li>
              <li>
                • <strong>Parameter tuning:</strong> Experiment with different
                thresholds, K values, and kernel sizes
              </li>
              <li>
                • <strong>Post-processing:</strong> Use morphological operations
                to refine segmentation results
              </li>
              <li>
                • <strong>Validation:</strong> Visually inspect results and
                compare different methods
              </li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
