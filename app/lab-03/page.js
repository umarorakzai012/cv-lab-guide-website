import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { Waves, Code, CheckCircle } from "lucide-react";

export default function Lab03() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Lab 03: Filtering & Fourier Transforms
        </h1>
        <p className="text-slate-600 text-lg">
          Linear/non-linear filtering, sampling techniques, and frequency domain
          analysis
        </p>
      </div>

      <Tabs defaultValue="concepts" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="concepts">
            <Waves className="w-4 h-4 mr-2" />
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
          <ConceptCard title="Linear Filtering">
            <div className="space-y-4 text-slate-700">
              <p>
                Linear filtering involves modifying an image by applying a
                linear operation called a filter or kernel. The filter is
                convolved with the image, sliding over it and computing weighted
                sums at each position.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  Key Concepts:
                </p>
                <ul className="space-y-2 text-blue-800">
                  <li>
                    <strong>Filter Kernel:</strong> A small matrix (e.g., 3×3,
                    5×5) defining weights for convolution
                  </li>
                  <li>
                    <strong>Convolution:</strong> Slide kernel over image,
                    multiply and sum corresponding values
                  </li>
                  <li>
                    <strong>Border Handling:</strong> Methods to handle pixels
                    at image edges
                  </li>
                </ul>
              </div>

              <div className="grid md:grid-cols-3 gap-4 mt-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Smoothing Filters
                  </h4>
                  <p className="text-sm text-green-800">
                    Reduce noise and blur images (Mean, Gaussian)
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Edge Detection
                  </h4>
                  <p className="text-sm text-purple-800">
                    Highlight boundaries (Sobel, Canny, Laplacian)
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900 mb-2">
                    Sharpening
                  </h4>
                  <p className="text-sm text-orange-800">
                    Enhance edges and details (Laplacian, Unsharp Mask)
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Smoothing Filters">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="mean">
                <AccordionTrigger>Mean (Box) Filter</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3">
                    <p className="text-slate-700">
                      Replaces each pixel with the average of its neighborhood.
                      Simple but effective for noise reduction.
                    </p>
                    <div className="bg-slate-100 p-3 rounded">
                      <p className="text-sm font-mono">
                        Kernel (3×3): [1/9, 1/9, 1/9]
                        <br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1/9,
                        1/9, 1/9]
                        <br />
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1/9,
                        1/9, 1/9]
                      </p>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="gaussian">
                <AccordionTrigger>Gaussian Filter</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3">
                    <p className="text-slate-700">
                      Uses a Gaussian-shaped kernel that emphasizes the central
                      pixel and diminishes influence of distant pixels. Provides
                      smoother results than box filter.
                    </p>
                    <div className="bg-blue-50 p-3 rounded">
                      <p className="text-sm text-blue-800">
                        <strong>Parameters:</strong> Kernel size (must be odd)
                        and sigma (standard deviation)
                      </p>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </ConceptCard>

          <ConceptCard title="Edge Detection Filters">
            <div className="space-y-4 text-slate-700">
              <p>
                Edge detection identifies boundaries between objects by finding
                rapid changes in pixel intensity.
              </p>

              <div className="space-y-3">
                <div className="border-l-4 border-blue-500 pl-4 py-2">
                  <h4 className="font-semibold">Sobel Filter</h4>
                  <p className="text-sm">
                    Calculates gradient using two 3×3 kernels for horizontal and
                    vertical edges. Magnitude = √(Gx² + Gy²)
                  </p>
                </div>

                <div className="border-l-4 border-green-500 pl-4 py-2">
                  <h4 className="font-semibold">Canny Edge Detector</h4>
                  <p className="text-sm">
                    Multi-stage: Gaussian smoothing → Gradient calculation →
                    Non-maximum suppression → Hysteresis thresholding
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4 py-2">
                  <h4 className="font-semibold">Laplacian of Gaussian (LoG)</h4>
                  <p className="text-sm">
                    Applies Gaussian smoothing then Laplacian to find areas of
                    rapid intensity change. Zero-crossings indicate edges.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Sharpening Filters">
            <div className="space-y-4 text-slate-700">
              <p>
                Sharpening enhances fine details and edges by emphasizing
                high-frequency components.
              </p>

              <Accordion type="single" collapsible>
                <AccordionItem value="laplacian">
                  <AccordionTrigger>Laplacian Filter</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">
                      Calculates the second derivative, highlighting rapid
                      intensity changes:
                    </p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      Sharpened = Original + Laplacian(Original)
                    </code>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="unsharp">
                  <AccordionTrigger>Unsharp Masking</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">
                      Subtracts blurred version from original to enhance edges:
                    </p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      Sharpened = 2×Original - GaussianBlur(Original)
                    </code>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="highboost">
                  <AccordionTrigger>High-Boost Filtering</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">
                      Extension of unsharp masking with adjustable strength:
                    </p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      Sharpened = A×Original - Blurred
                      <br />
                      where A controls sharpening strength
                    </code>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>

          <ConceptCard title="Non-Linear Filtering">
            <div className="space-y-4 text-slate-700">
              <p>
                Non-linear filters use non-linear operations on pixel values,
                making output dependent on pixel ranking or other non-linear
                criteria.
              </p>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-red-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-red-900 mb-2">
                    Median Filter
                  </h4>
                  <p className="text-sm text-red-800">
                    Replaces pixel with median of neighborhood. Excellent for
                    removing salt-and-pepper noise while preserving edges.
                  </p>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    Min/Max Filters
                  </h4>
                  <p className="text-sm text-blue-800">
                    Min filter: erosion operation. Max filter: dilation
                    operation. Used in morphological processing.
                  </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Bilateral Filter
                  </h4>
                  <p className="text-sm text-green-800">
                    Smooths image while preserving edges by considering both
                    spatial distance and intensity similarity.
                  </p>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Adaptive Median
                  </h4>
                  <p className="text-sm text-purple-800">
                    Adapts kernel size based on local statistics. Better noise
                    removal with less blurring.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Border Handling Methods">
            <div className="space-y-4 text-slate-700">
              <p>
                When applying filters, edge pixels require special handling
                since the kernel extends beyond image boundaries.
              </p>

              <div className="space-y-3">
                <div className="bg-slate-100 p-3 rounded">
                  <h4 className="font-semibold mb-1">BORDER_REPLICATE</h4>
                  <p className="text-sm">
                    Replicates border pixels: aaaaaa|abcdefgh|hhhhhhh
                  </p>
                </div>

                <div className="bg-slate-100 p-3 rounded">
                  <h4 className="font-semibold mb-1">BORDER_CONSTANT</h4>
                  <p className="text-sm">
                    Pads with constant value (usually 0):
                    000000|abcdefgh|0000000
                  </p>
                </div>

                <div className="bg-slate-100 p-3 rounded">
                  <h4 className="font-semibold mb-1">BORDER_REFLECT</h4>
                  <p className="text-sm">
                    Reflects border pixels: fedcba|abcdefgh|hgfedcb
                  </p>
                </div>

                <div className="bg-slate-100 p-3 rounded">
                  <h4 className="font-semibold mb-1">BORDER_WRAP</h4>
                  <p className="text-sm">
                    Wraps around like torus: cdefgh|abcdefgh|abcdefg
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="2D Fourier Transform">
            <div className="space-y-4 text-slate-700">
              <p>
                The 2D Fourier Transform decomposes an image into sinusoidal
                components, converting from spatial domain to frequency domain.
              </p>

              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg">
                <p className="font-semibold text-slate-900 mb-2">
                  Key Concepts:
                </p>
                <ul className="space-y-2 text-slate-800">
                  <li>
                    <strong>Spatial Domain:</strong> Image as pixel intensities
                    (what we see)
                  </li>
                  <li>
                    <strong>Frequency Domain:</strong> Image as sum of
                    sinusoidal waves
                  </li>
                  <li>
                    <strong>Low Frequencies:</strong> Smooth regions, overall
                    structure
                  </li>
                  <li>
                    <strong>High Frequencies:</strong> Edges, fine details,
                    noise
                  </li>
                </ul>
              </div>

              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded mt-4">
                <p className="font-semibold text-yellow-900 mb-2">
                  Applications:
                </p>
                <ul className="text-sm text-yellow-800 space-y-1">
                  <li>
                    • <strong>Filtering:</strong> Low-pass (smoothing),
                    high-pass (edge enhancement)
                  </li>
                  <li>
                    • <strong>Compression:</strong> Discard high frequencies for
                    smaller file size
                  </li>
                  <li>
                    • <strong>Feature Extraction:</strong> Analyze frequency
                    patterns
                  </li>
                  <li>
                    • <strong>Image Registration:</strong> Phase correlation for
                    alignment
                  </li>
                </ul>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Sampling in 1D and 2D">
            <div className="space-y-4 text-slate-700">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    1D Sampling
                  </h4>
                  <p className="text-sm text-blue-800">
                    Capturing discrete samples from a continuous signal at
                    regular intervals (e.g., audio sampling, temperature
                    readings).
                  </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    2D Sampling
                  </h4>
                  <p className="text-sm text-green-800">
                    Selecting discrete pixels from continuous 2D images. Each
                    pixel represents intensity/color at that location.
                  </p>
                </div>
              </div>

              <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                <p className="font-semibold text-red-900 mb-2">
                  Nyquist-Shannon Theorem:
                </p>
                <p className="text-sm text-red-800">
                  Sampling rate must be at least twice the highest frequency to
                  avoid aliasing (information loss).
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Hybrid Images">
            <div className="space-y-4 text-slate-700">
              <p>
                Hybrid images combine high-frequency details from one image with
                low-frequency features from another, creating an image that
                appears different at varying distances.
              </p>

              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
                <p className="font-semibold mb-2">How It Works:</p>
                <ol className="list-decimal list-inside space-y-2 text-sm">
                  <li>
                    Extract high frequencies from first image (apply high-pass
                    filter)
                  </li>
                  <li>
                    Extract low frequencies from second image (apply low-pass
                    filter)
                  </li>
                  <li>Combine both filtered images</li>
                  <li>
                    Result: Close viewing shows high-freq image, distant viewing
                    shows low-freq image
                  </li>
                </ol>
              </div>

              <p className="text-sm">
                <strong>Example:</strong> Hybrid of Einstein (low-freq) and
                Marilyn Monroe (high-freq) appears as Einstein from distance,
                Marilyn up close.
              </p>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="Mean (Box) Filter">
            <div className="space-y-4">
              <p className="text-slate-700">
                Simple averaging filter that reduces noise by replacing each
                pixel with the mean of its neighborhood.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Apply mean filter with 5x5 kernel
mean_filtered = cv2.blur(image, (5, 5))

# Alternative: using custom kernel
kernel = np.ones((5, 5), dtype=np.float32) / 25
custom_filtered = cv2.filter2D(image, -1, kernel)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Mean Filtered', mean_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Gaussian Blur">
            <div className="space-y-4">
              <p className="text-slate-700">
                Gaussian filter provides smoother results than box filter by
                using a Gaussian-weighted kernel.
              </p>
              <CodeBlock
                code={`import cv2

# Load image
image = cv2.imread('image.jpg')

# Apply Gaussian blur
# Parameters: image, kernel size (must be odd), sigma (0 = auto-calculate)
gaussian_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# With specific sigma value
gaussian_sigma2 = cv2.GaussianBlur(image, (5, 5), 2.0)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Gaussian (sigma=auto)', gaussian_blurred)
cv2.imshow('Gaussian (sigma=2)', gaussian_sigma2)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-blue-50 p-4 rounded">
                <p className="text-sm text-blue-800">
                  <strong>Tip:</strong> Larger kernel sizes and higher sigma
                  values produce stronger blur effects.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Sobel Edge Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                Sobel operator calculates image gradient to detect edges in
                horizontal and vertical directions.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate gradients in x and y directions
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate gradient magnitude
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude = np.uint8(gradient_magnitude)

# Calculate gradient direction
gradient_direction = np.arctan2(sobel_y, sobel_x)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Sobel X', np.uint8(np.absolute(sobel_x)))
cv2.imshow('Sobel Y', np.uint8(np.absolute(sobel_y)))
cv2.imshow('Gradient Magnitude', gradient_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Canny Edge Detection">
            <div className="space-y-4">
              <p className="text-slate-700">
                Canny is a multi-stage edge detection algorithm known for
                accuracy and noise suppression.
              </p>
              <CodeBlock
                code={`import cv2

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
# Parameters: image, low threshold, high threshold
edges = cv2.Canny(image, 100, 200)

# With different thresholds
edges_sensitive = cv2.Canny(image, 50, 150)
edges_conservative = cv2.Canny(image, 150, 250)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Canny Edges (100,200)', edges)
cv2.imshow('Sensitive (50,150)', edges_sensitive)
cv2.imshow('Conservative (150,250)', edges_conservative)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                <p className="text-sm text-yellow-800">
                  <strong>Guideline:</strong> High threshold should be 2-3× the
                  low threshold. Adjust based on image noise level.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Laplacian Sharpening">
            <div className="space-y-4">
              <p className="text-slate-700">
                Laplacian filter enhances edges by adding the second derivative
                to the original image.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Apply Laplacian filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Sharpen by adding Laplacian to original
sharpened = cv2.add(image, np.uint8(np.absolute(laplacian)))

# Alternative: convert to grayscale first
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lap_gray = cv2.Laplacian(gray, cv2.CV_64F)
sharp_gray = np.clip(gray + lap_gray, 0, 255).astype(np.uint8)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Laplacian', np.uint8(np.absolute(laplacian)))
cv2.imshow('Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Unsharp Masking">
            <div className="space-y-4">
              <p className="text-slate-700">
                Unsharp masking enhances details by subtracting a blurred
                version from the original.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Create blurred version
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Unsharp masking: 2*original - blurred
sharpened = cv2.addWeighted(image, 2.0, blurred, -1.0, 0)

# Alternative with adjustable strength
alpha = 1.5  # Amount of original
beta = -0.5  # Amount of blur (negative)
sharpened_custom = cv2.addWeighted(image, alpha, blurred, beta, 0)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.imshow('Unsharp Masked', sharpened)
cv2.imshow('Custom Strength', sharpened_custom)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Median Filter">
            <div className="space-y-4">
              <p className="text-slate-700">
                Median filter is excellent for removing salt-and-pepper noise
                while preserving edges.
              </p>
              <CodeBlock
                code={`import cv2

# Load image
image = cv2.imread('image.jpg')

# Apply median filter (kernel size must be odd)
median_filtered = cv2.medianBlur(image, 5)

# Different kernel sizes
median_3x3 = cv2.medianBlur(image, 3)
median_7x7 = cv2.medianBlur(image, 7)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Median 3x3', median_3x3)
cv2.imshow('Median 5x5', median_filtered)
cv2.imshow('Median 7x7', median_7x7)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Bilateral Filter">
            <div className="space-y-4">
              <p className="text-slate-700">
                Bilateral filter smooths while preserving edges by considering
                spatial and intensity similarity.
              </p>
              <CodeBlock
                code={`import cv2

# Load image
image = cv2.imread('image.jpg')

# Apply bilateral filter
# Parameters: image, d (diameter), sigmaColor, sigmaSpace
bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Different parameters
bilateral_strong = cv2.bilateralFilter(image, 9, 150, 150)
bilateral_weak = cv2.bilateralFilter(image, 9, 25, 25)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Bilateral (75,75)', bilateral)
cv2.imshow('Strong (150,150)', bilateral_strong)
cv2.imshow('Weak (25,25)', bilateral_weak)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-green-50 p-4 rounded">
                <p className="text-sm text-green-800">
                  <strong>Parameters:</strong> Higher sigmaColor = more colors
                  mixed. Higher sigmaSpace = larger area affected.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Border Handling Comparison">
            <div className="space-y-4">
              <p className="text-slate-700">
                Compare different border handling methods when applying filters.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a simple kernel
kernel = np.array([[0, 1, 0],
                   [1, 5, 1],
                   [0, 1, 0]], dtype=np.float32)
kernel /= kernel.sum()

# Apply with different border types
replicate = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
constant = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
reflect = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
wrap = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)

# Display results
cv2.imshow('Original', image)
cv2.imshow('REPLICATE', replicate)
cv2.imshow('CONSTANT', constant)
cv2.imshow('REFLECT', reflect)
cv2.imshow('WRAP', wrap)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="2D Fourier Transform - Forward">
            <div className="space-y-4">
              <p className="text-slate-700">
                Convert image from spatial domain to frequency domain using 2D
                FFT.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform 2D Fourier Transform
fourier_transform = np.fft.fft2(image)

# Shift zero frequency components to center
fourier_shifted = np.fft.fftshift(fourier_transform)

# Calculate magnitude spectrum (log scale for visualization)
magnitude_spectrum = np.log(np.abs(fourier_shifted) + 1)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (log-scaled)')
plt.axis('off')

plt.tight_layout()
plt.show()`}
              />
              <div className="bg-blue-50 p-4 rounded">
                <p className="text-sm text-blue-800">
                  <strong>Understanding the spectrum:</strong> Bright center =
                  low frequencies (smooth regions). Outer areas = high
                  frequencies (edges, details).
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="2D Fourier Transform - Inverse">
            <div className="space-y-4">
              <p className="text-slate-700">
                Convert from frequency domain back to spatial domain to
                reconstruct the image.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and apply forward transform
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(fourier)

# Perform inverse transform
# First, shift back
fourier_ishift = np.fft.ifftshift(fourier_shifted)

# Apply inverse FFT
reconstructed = np.fft.ifft2(fourier_ishift)

# Take the real part (imaginary part should be negligible)
reconstructed_image = np.real(reconstructed)

# Convert to uint8 for display
reconstructed_image = np.uint8(reconstructed_image)

# Display comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Verify reconstruction accuracy
difference = np.abs(image - reconstructed_image)
print(f"Maximum difference: {np.max(difference)}")
print(f"Mean difference: {np.mean(difference):.4f}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Low-Pass Filtering in Frequency Domain">
            <div className="space-y-4">
              <p className="text-slate-700">
                Apply low-pass filter in frequency domain to smooth image by
                removing high frequencies.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(fourier)

# Create low-pass filter mask
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
cutoff_frequency = 30  # Adjust for stronger/weaker filtering

# Initialize mask with zeros
mask = np.zeros((rows, cols), dtype=np.uint8)

# Create circular low-pass filter
for i in range(rows):
    for j in range(cols):
        # Calculate distance from center
        distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
        if distance <= cutoff_frequency:
            mask[i, j] = 1

# Alternative: rectangular mask (faster)
mask_rect = np.zeros((rows, cols), dtype=np.uint8)
mask_rect[center_row - cutoff_frequency:center_row + cutoff_frequency + 1,
          center_col - cutoff_frequency:center_col + cutoff_frequency + 1] = 1

# Apply mask to Fourier Transform
filtered_fourier = fourier_shifted * mask_rect

# Inverse transform
filtered_ishift = np.fft.ifftshift(filtered_fourier)
filtered_image = np.fft.ifft2(filtered_ishift)
filtered_image = np.abs(filtered_image)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.log(np.abs(fourier_shifted) + 1), cmap='gray')
axes[0, 1].set_title('Frequency Spectrum')
axes[0, 1].axis('off')

axes[1, 0].imshow(mask_rect, cmap='gray')
axes[1, 0].set_title('Low-Pass Filter Mask')
axes[1, 0].axis('off')

axes[1, 1].imshow(filtered_image, cmap='gray')
axes[1, 1].set_title('Low-Pass Filtered Image')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()`}
              />
              <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                <p className="text-sm text-purple-800">
                  <strong>Effect:</strong> Smaller cutoff frequency = stronger
                  smoothing. This removes fine details and noise.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="High-Pass Filtering in Frequency Domain">
            <div className="space-y-4">
              <p className="text-slate-700">
                Apply high-pass filter to enhance edges by removing low
                frequencies.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(fourier)

# Create high-pass filter mask (inverse of low-pass)
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
cutoff_frequency = 30

# Initialize mask with ones
mask = np.ones((rows, cols), dtype=np.uint8)

# Block low frequencies (center region)
mask[center_row - cutoff_frequency:center_row + cutoff_frequency + 1,
     center_col - cutoff_frequency:center_col + cutoff_frequency + 1] = 0

# Apply mask
filtered_fourier = fourier_shifted * mask

# Inverse transform
filtered_ishift = np.fft.ifftshift(filtered_fourier)
filtered_image = np.fft.ifft2(filtered_ishift)
filtered_image = np.abs(filtered_image)

# Normalize for display
filtered_image = np.uint8(255 * filtered_image / np.max(filtered_image))

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('High-Pass Filter Mask')
axes[1].axis('off')

axes[2].imshow(filtered_image, cmap='gray')
axes[2].set_title('High-Pass Filtered (Edges Enhanced)')
axes[2].axis('off')

plt.tight_layout()
plt.show()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Compression using FFT">
            <div className="space-y-4">
              <p className="text-slate-700">
                Compress image by discarding small frequency components in the
                Fourier domain.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(fourier)

# Set threshold to retain only significant components
threshold = 1000  # Adjust based on image
filtered_fourier = fourier_shifted.copy()
filtered_fourier[np.abs(filtered_fourier) < threshold] = 0

# Calculate compression ratio
total_coeffs = filtered_fourier.size
retained_coeffs = np.count_nonzero(filtered_fourier)
compression_ratio = 100 * (1 - retained_coeffs / total_coeffs)

# Inverse transform to get compressed image
compressed_ishift = np.fft.ifftshift(filtered_fourier)
compressed_image = np.fft.ifft2(compressed_ishift)
compressed_image = np.abs(compressed_image)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(np.log(np.abs(filtered_fourier) + 1), cmap='gray')
axes[1].set_title(f'Compressed Spectrum\\n{compression_ratio:.1f}% compressed')
axes[1].axis('off')

axes[2].imshow(compressed_image, cmap='gray')
axes[2].set_title('Compressed Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"Compression ratio: {compression_ratio:.2f}%")
print(f"Retained coefficients: {retained_coeffs}/{total_coeffs}")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Hybrid Images">
            <div className="space-y-4">
              <p className="text-slate-700">
                Create hybrid images by combining high frequencies from one
                image with low frequencies from another.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images (should be aligned and same size)
image1 = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('marilyn.jpg', cv2.IMREAD_GRAYSCALE)

# Resize to same dimensions
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Extract low frequencies from first image (Gaussian blur)
low_freq = cv2.GaussianBlur(image1, (25, 25), 10)

# Extract high frequencies from second image
# Method 1: Subtract Gaussian blur
blurred2 = cv2.GaussianBlur(image2, (25, 25), 10)
high_freq = image2 - blurred2

# Method 2: Using Fourier Transform for high-pass
fourier2 = np.fft.fft2(image2)
fourier_shifted = np.fft.fftshift(fourier2)

# Create high-pass mask
rows, cols = image2.shape
center_row, center_col = rows // 2, cols // 2
kernel_size = 30

mask = np.ones((rows, cols), dtype=np.uint8)
mask[center_row - kernel_size:center_row + kernel_size + 1,
     center_col - kernel_size:center_col + kernel_size + 1] = 0

# Apply high-pass filter
filtered_fourier = fourier_shifted * mask
high_freq_fft = np.fft.ifft2(np.fft.ifftshift(filtered_fourier))
high_freq_fft = np.real(high_freq_fft)

# Combine low and high frequencies
hybrid = low_freq + high_freq

# Normalize to valid range
hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image1, cmap='gray')
axes[0, 0].set_title('Image 1 (Einstein)')
axes[0, 0].axis('off')

axes[0, 1].imshow(low_freq, cmap='gray')
axes[0, 1].set_title('Low Frequencies\\n(view from far)')
axes[0, 1].axis('off')

axes[0, 2].imshow(image2, cmap='gray')
axes[0, 2].set_title('Image 2 (Marilyn)')
axes[0, 2].axis('off')

axes[1, 0].imshow(high_freq + 128, cmap='gray')  # Add 128 for visibility
axes[1, 0].set_title('High Frequencies\\n(view from close)')
axes[1, 0].axis('off')

axes[1, 1].imshow(hybrid, cmap='gray')
axes[1, 1].set_title('Hybrid Image\\n(Einstein far, Marilyn close)')
axes[1, 1].axis('off')

axes[1, 2].imshow(hybrid, cmap='gray')
axes[1, 2].set_title('Hybrid Image (zoomed out)')
axes[1, 2].axis('off')
# Zoom out effect
axes[1, 2].set_xlim(cols//4, 3*cols//4)
axes[1, 2].set_ylim(3*rows//4, rows//4)

plt.tight_layout()
plt.show()

cv2.imwrite('hybrid_image.jpg', hybrid)
print("Hybrid image saved!")`}
              />
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded">
                <p className="font-semibold mb-2">Tips for Best Results:</p>
                <ul className="text-sm space-y-1">
                  <li>• Use aligned images with similar composition</li>
                  <li>• Adjust Gaussian sigma (10-20) for low-pass strength</li>
                  <li>
                    • Adjust cutoff frequency (20-40) for high-pass strength
                  </li>
                  <li>• Test different combinations to find optimal balance</li>
                </ul>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="1D and 2D Sampling Visualization">
            <div className="space-y-4">
              <p className="text-slate-700">
                Visualize the concept of sampling continuous signals at discrete
                intervals.
              </p>
              <CodeBlock
                code={`import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1D Sampling Example
# Create continuous signal
time = np.linspace(0, 10, 1000)
continuous_signal = np.sin(2 * np.pi * time) + 0.5 * np.sin(4 * np.pi * time)

# Sample at discrete intervals
sampling_rate = 20  # samples per second
sample_indices = np.arange(0, len(time), len(time) // (10 * sampling_rate))
sampled_signal = continuous_signal[sample_indices]
sampled_time = time[sample_indices]

# Plot 1D sampling
plt.figure(figsize=(12, 4))
plt.plot(time, continuous_signal, 'b-', label='Continuous Signal', linewidth=2)
plt.stem(sampled_time, sampled_signal, linefmt='r-', markerfmt='ro', 
         basefmt=' ', label='Sampled Points')
plt.title('1D Sampling: Continuous vs Discrete')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2D Sampling Example (Image)
# Load high-resolution image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Downsample (reduce sampling rate)
downsampled_2x = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
downsampled_4x = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

# Upsample back to original size for comparison
upsampled_2x = cv2.resize(downsampled_2x, (image.shape[1], image.shape[0]))
upsampled_4x = cv2.resize(downsampled_4x, (image.shape[1], image.shape[0]))

# Display 2D sampling results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original (Full Sampling)')
axes[0, 0].axis('off')

axes[0, 1].imshow(downsampled_2x, cmap='gray')
axes[0, 1].set_title('Downsampled 2× (Half Sampling)')
axes[0, 1].axis('off')

axes[0, 2].imshow(downsampled_4x, cmap='gray')
axes[0, 2].set_title('Downsampled 4× (Quarter Sampling)')
axes[0, 2].axis('off')

axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title('Original Resolution')
axes[1, 0].axis('off')

axes[1, 1].imshow(upsampled_2x, cmap='gray')
axes[1, 1].set_title('Reconstructed from 2× Downsampling')
axes[1, 1].axis('off')

axes[1, 2].imshow(upsampled_4x, cmap='gray')
axes[1, 2].set_title('Reconstructed from 4× Downsampling\\n(Aliasing visible)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print(f"Original size: {image.shape}")
print(f"2× downsampled: {downsampled_2x.shape}")
print(f"4× downsampled: {downsampled_4x.shape}")`}
              />
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-6 mt-6">
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 border-2 border-cyan-200 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-cyan-900 mb-4">Lab Tasks</h2>
            <p className="text-cyan-800 mb-4">
              Apply filtering and Fourier transform techniques to real-world
              image processing challenges.
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="task1">
              <AccordionTrigger className="text-lg font-semibold">
                Task 1: Implement All Smoothing and Sharpening Filters
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <h3 className="font-semibold text-blue-900 mb-2">
                    Objective:
                  </h3>
                  <p className="text-blue-800">
                    Implement and compare various smoothing and sharpening
                    filters on the same image.
                  </p>
                </div>

                <div className="space-y-3 text-slate-700">
                  <h4 className="font-semibold">Requirements:</h4>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li>
                      <strong>Smoothing:</strong> Mean filter, Gaussian blur
                      (σ=1 and σ=3)
                    </li>
                    <li>
                      <strong>Sharpening:</strong> Laplacian, Unsharp masking,
                      High-boost filtering
                    </li>
                    <li>
                      <strong>Edge Detection:</strong> Sobel, Canny (with 3
                      different threshold pairs)
                    </li>
                    <li>
                      Display all results in a grid using matplotlib subplots
                    </li>
                    <li>Add titles showing filter parameters</li>
                  </ul>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smoothing filters
mean_filter = cv2.blur(image, (5, 5))
gaussian_sigma1 = cv2.GaussianBlur(image, (5, 5), 1)
gaussian_sigma3 = cv2.GaussianBlur(image, (5, 5), 3)

# Sharpening filters
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened_lap = np.clip(gray + laplacian, 0, 255).astype(np.uint8)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
unsharp = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)

high_boost = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)

# Edge detection
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.uint8(np.sqrt(sobel_x**2 + sobel_y**2))

canny1 = cv2.Canny(gray, 50, 150)
canny2 = cv2.Canny(gray, 100, 200)
canny3 = cv2.Canny(gray, 150, 250)

# Display in grid
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 1].imshow(cv2.cvtColor(mean_filter, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Mean Filter 5×5')
axes[0, 2].imshow(cv2.cvtColor(gaussian_sigma1, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Gaussian σ=1')
axes[0, 3].imshow(cv2.cvtColor(gaussian_sigma3, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('Gaussian σ=3')

axes[1, 0].imshow(sharpened_lap, cmap='gray')
axes[1, 0].set_title('Laplacian Sharpened')
axes[1, 1].imshow(unsharp, cmap='gray')
axes[1, 1].set_title('Unsharp Masking')
axes[1, 2].imshow(high_boost, cmap='gray')
axes[1, 2].set_title('High-Boost Filter')
axes[1, 3].imshow(sobel_combined, cmap='gray')
axes[1, 3].set_title('Sobel Edges')

axes[2, 0].imshow(gray, cmap='gray')
axes[2, 0].set_title('Grayscale Original')
axes[2, 1].imshow(canny1, cmap='gray')
axes[2, 1].set_title('Canny (50,150)')
axes[2, 2].imshow(canny2, cmap='gray')
axes[2, 2].set_title('Canny (100,200)')
axes[2, 3].imshow(canny3, cmap='gray')
axes[2, 3].set_title('Canny (150,250)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task2">
              <AccordionTrigger className="text-lg font-semibold">
                Task 2: Non-Linear Filters Comparison
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                  <h3 className="font-semibold text-green-900 mb-2">
                    Objective:
                  </h3>
                  <p className="text-green-800">
                    Compare non-linear filtering techniques on a noisy image.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg')

# Add salt-and-pepper noise
noisy = image.copy()
salt_pepper_ratio = 0.02
num_salt = int(salt_pepper_ratio * image.size / 2)

# Add salt noise (white pixels)
coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
noisy[coords[0], coords[1]] = 255

# Add pepper noise (black pixels)
coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
noisy[coords[0], coords[1]] = 0

# Apply non-linear filters
median_3 = cv2.medianBlur(noisy, 3)
median_5 = cv2.medianBlur(noisy, 5)
median_7 = cv2.medianBlur(noisy, 7)

bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)

# Min and max filters (morphological operations)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
min_filter = cv2.erode(noisy, kernel)
max_filter = cv2.dilate(noisy, kernel)

# Display results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Noisy (Salt & Pepper)')
axes[0, 2].imshow(cv2.cvtColor(median_3, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Median 3×3')
axes[0, 3].imshow(cv2.cvtColor(median_5, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('Median 5×5')

axes[1, 0].imshow(cv2.cvtColor(median_7, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Median 7×7')
axes[1, 1].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Bilateral Filter')
axes[1, 2].imshow(cv2.cvtColor(min_filter, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Min Filter (Erosion)')
axes[1, 3].imshow(cv2.cvtColor(max_filter, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('Max Filter (Dilation)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task3">
              <AccordionTrigger className="text-lg font-semibold">
                Task 3: Fourier Transform Applications
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                  <h3 className="font-semibold text-purple-900 mb-2">
                    Objective:
                  </h3>
                  <p className="text-purple-800">
                    Explore various applications of 2D Fourier Transform.
                  </p>
                </div>

                <div className="space-y-3 text-slate-700">
                  <h4 className="font-semibold">Tasks to Complete:</h4>
                  <ol className="list-decimal list-inside space-y-2 ml-4">
                    <li>
                      Calculate 1D FFT of a signal and visualize magnitude/phase
                      spectra
                    </li>
                    <li>
                      Apply 2D FFT to an image and display magnitude spectrum
                    </li>
                    <li>Implement high-pass filter to emphasize edges</li>
                    <li>
                      Perform image compression by thresholding frequency
                      components
                    </li>
                    <li>Compare results with different cutoff frequencies</li>
                  </ol>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 2D Fourier Transform
fourier = np.fft.fft2(image)
fourier_shifted = np.fft.fftshift(fourier)
magnitude_spectrum = np.log(np.abs(fourier_shifted) + 1)

# High-pass filtering
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask_highpass = np.ones((rows, cols), dtype=np.uint8)
r = 30
mask_highpass[crow-r:crow+r, ccol-r:ccol+r] = 0

highpass_filtered = fourier_shifted * mask_highpass
highpass_image = np.fft.ifft2(np.fft.ifftshift(highpass_filtered))
highpass_image = np.abs(highpass_image)

# Image compression
threshold = np.percentile(np.abs(fourier_shifted), 90)
compressed_fourier = fourier_shifted.copy()
compressed_fourier[np.abs(compressed_fourier) < threshold] = 0

compressed_image = np.fft.ifft2(np.fft.ifftshift(compressed_fourier))
compressed_image = np.abs(compressed_image)

compression_ratio = 100 * (1 - np.count_nonzero(compressed_fourier) / compressed_fourier.size)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(magnitude_spectrum, cmap='gray')
axes[0, 1].set_title('Magnitude Spectrum')
axes[0, 2].imshow(mask_highpass, cmap='gray')
axes[0, 2].set_title('High-Pass Filter Mask')

axes[1, 0].imshow(highpass_image, cmap='gray')
axes[1, 0].set_title('High-Pass Filtered (Edges)')
axes[1, 1].imshow(compressed_image, cmap='gray')
axes[1, 1].set_title(f'Compressed ({compression_ratio:.1f}%)')
axes[1, 2].imshow(np.log(np.abs(compressed_fourier) + 1), cmap='gray')
axes[1, 2].set_title('Compressed Spectrum')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task4">
              <AccordionTrigger className="text-lg font-semibold">
                Task 4: Create Your Own Hybrid Images
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
                  <h3 className="font-semibold text-orange-900 mb-2">
                    Objective:
                  </h3>
                  <p className="text-orange-800">
                    Create hybrid images with different combinations and analyze
                    the visual illusion effect.
                  </p>
                </div>

                <div className="space-y-3 text-slate-700">
                  <h4 className="font-semibold">Requirements:</h4>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li>
                      Select two images with similar composition (faces,
                      objects, etc.)
                    </li>
                    <li>
                      Experiment with different high-pass and low-pass filter
                      parameters
                    </li>
                    <li>
                      Create at least 3 hybrid images with varying filter
                      strengths
                    </li>
                    <li>
                      Test how changing sigma affects the perception distance
                    </li>
                    <li>Document which combination works best and why</li>
                  </ul>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and align two images
img1 = cv2.imread('face1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('face2.jpg', cv2.IMREAD_GRAYSCALE)

# Resize to match
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Function to create hybrid image
def create_hybrid(img1, img2, sigma_low, sigma_high):
    # Low frequencies from img1
    low_freq = cv2.GaussianBlur(img1, (0, 0), sigma_low)
    
    # High frequencies from img2
    blurred2 = cv2.GaussianBlur(img2, (0, 0), sigma_high)
    high_freq = img2 - blurred2
    
    # Combine
    hybrid = low_freq + high_freq
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
    
    return hybrid, low_freq, high_freq

# Experiment with different parameters
params = [
    (5, 5, "Weak Effect"),
    (10, 3, "Medium Effect"),
    (15, 2, "Strong Effect")
]

fig, axes = plt.subplots(len(params), 4, figsize=(16, 12))

for i, (sigma_low, sigma_high, label) in enumerate(params):
    hybrid, low, high = create_hybrid(img1, img2, sigma_low, sigma_high)
    
    axes[i, 0].imshow(img1, cmap='gray')
    axes[i, 0].set_title(f'Image 1 (Low-freq source)')
    
    axes[i, 1].imshow(img2, cmap='gray')
    axes[i, 1].set_title(f'Image 2 (High-freq source)')
    
    axes[i, 2].imshow(hybrid, cmap='gray')
    axes[i, 2].set_title(f'Hybrid: {label}\\nσ_low={sigma_low}, σ_high={sigma_high}')
    
    # Create scaled down version to simulate viewing from distance
    small = cv2.resize(hybrid, None, fx=0.3, fy=0.3)
    scaled_back = cv2.resize(small, (hybrid.shape[1], hybrid.shape[0]))
    axes[i, 3].imshow(scaled_back, cmap='gray')
    axes[i, 3].set_title('Simulated Distance View')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

# Save best hybrid
best_hybrid, _, _ = create_hybrid(img1, img2, 10, 3)
cv2.imwrite('my_hybrid_image.jpg', best_hybrid)
print("Best hybrid image saved!")

# Analysis
print("\\nHybrid Image Analysis:")
print("=" * 50)
print(f"Parameter Set 1 (σ_low=5, σ_high=5): Weak separation")
print(f"  - Both images visible at most distances")
print(f"Parameter Set 2 (σ_low=10, σ_high=3): Optimal balance")
print(f"  - Clear switch between close/far viewing")
print(f"Parameter Set 3 (σ_low=15, σ_high=2): Strong effect")
print(f"  - Very distinct images at different distances")
print(f"  - May lose some details")`}
                />

                <div className="bg-blue-50 p-4 rounded mt-4">
                  <p className="font-semibold text-blue-900 mb-2">
                    Expected Observations:
                  </p>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>
                      • Higher σ_low = stronger blur, more dominant
                      low-frequency image from distance
                    </li>
                    <li>
                      • Lower σ_high = sharper edges, clearer high-frequency
                      image up close
                    </li>
                    <li>
                      • Best results when images have similar composition and
                      alignment
                    </li>
                    <li>
                      • Optimal viewing distance depends on filter parameters
                    </li>
                  </ul>
                </div>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task5">
              <AccordionTrigger className="text-lg font-semibold">
                Task 5: Border Handling Investigation
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-pink-50 border-l-4 border-pink-500 p-4 rounded">
                  <h3 className="font-semibold text-pink-900 mb-2">
                    Objective:
                  </h3>
                  <p className="text-pink-800">
                    Investigate how different border handling methods affect
                    filter results, especially at image edges.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Create a strong edge-detecting kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Apply with different border types
border_types = [
    (cv2.BORDER_REPLICATE, 'REPLICATE'),
    (cv2.BORDER_CONSTANT, 'CONSTANT'),
    (cv2.BORDER_REFLECT, 'REFLECT'),
    (cv2.BORDER_WRAP, 'WRAP'),
    (cv2.BORDER_REFLECT_101, 'REFLECT_101')
]

results = []
for border_type, name in border_types:
    filtered = cv2.filter2D(image, -1, kernel, borderType=border_type)
    results.append((filtered, name))

# Create visualization focusing on borders
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].add_patch(plt.Rectangle((0, 0), 50, image.shape[0], 
                                   fill=False, edgecolor='red', linewidth=2))

for idx, (result, name) in enumerate(results):
    ax = axes[(idx + 1) // 3, (idx + 1) % 3]
    ax.imshow(result, cmap='gray')
    ax.set_title(f'Border: {name}')
    # Highlight border region
    ax.add_patch(plt.Rectangle((0, 0), 50, result.shape[0], 
                               fill=False, edgecolor='red', linewidth=2))

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

# Zoom in on border region for detailed comparison
fig, axes = plt.subplots(1, len(border_types) + 1, figsize=(18, 3))

# Show original border region
border_region_orig = image[:100, :50]
axes[0].imshow(border_region_orig, cmap='gray')
axes[0].set_title('Original Border\\n(zoomed)')
axes[0].axis('off')

# Show filtered border regions
for idx, (result, name) in enumerate(results):
    border_region = result[:100, :50]
    axes[idx + 1].imshow(border_region, cmap='gray')
    axes[idx + 1].set_title(f'{name}\\n(zoomed)')
    axes[idx + 1].axis('off')

plt.tight_layout()
plt.show()

# Analysis
print("Border Handling Analysis:")
print("=" * 50)
for _, name in border_types:
    print(f"\\n{name}:")
    if name == 'REPLICATE':
        print("  - Replicates edge pixels")
        print("  - Good for natural images")
        print("  - May create visible edges")
    elif name == 'CONSTANT':
        print("  - Pads with zeros (black)")
        print("  - Creates dark borders")
        print("  - Good for detecting actual edges")
    elif name == 'REFLECT':
        print("  - Mirrors image at boundaries")
        print("  - Most natural looking")
        print("  - Minimizes artifacts")
    elif name == 'WRAP':
        print("  - Wraps around like a torus")
        print("  - Can create unnatural patterns")
        print("  - Good for periodic images")
    elif name == 'REFLECT_101':
        print("  - Reflects without repeating edge")
        print("  - Similar to REFLECT")
        print("  - Often best choice")`}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-l-4 border-green-500 p-6 rounded-lg">
            <h3 className="font-semibold text-green-900 mb-3">
              🎯 Learning Outcomes
            </h3>
            <div className="grid md:grid-cols-2 gap-4 text-green-800">
              <div>
                <h4 className="font-semibold mb-2">Linear Filtering:</h4>
                <ul className="text-sm space-y-1">
                  <li>• Understand convolution operation</li>
                  <li>• Apply smoothing, sharpening, and edge detection</li>
                  <li>• Choose appropriate kernel sizes</li>
                  <li>• Handle image borders correctly</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Non-Linear Filtering:</h4>
                <ul className="text-sm space-y-1">
                  <li>• Distinguish from linear filters</li>
                  <li>• Apply median filter for noise removal</li>
                  <li>• Use bilateral filter for edge preservation</li>
                  <li>• Understand morphological operations</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Fourier Transform:</h4>
                <ul className="text-sm space-y-1">
                  <li>• Convert spatial ↔ frequency domain</li>
                  <li>• Apply frequency domain filtering</li>
                  <li>• Implement image compression</li>
                  <li>• Understand frequency components</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Hybrid Images:</h4>
                <ul className="text-sm space-y-1">
                  <li>• Separate frequency components</li>
                  <li>• Combine high and low frequencies</li>
                  <li>• Create visual illusions</li>
                  <li>• Understand human visual perception</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg">
            <h3 className="font-semibold text-amber-900 mb-3">💡 Pro Tips</h3>
            <ul className="space-y-2 text-amber-800 text-sm">
              <li>
                • <strong>Kernel Size:</strong> Use odd sizes (3, 5, 7) for
                centered operations
              </li>
              <li>
                • <strong>Gaussian Blur:</strong> Larger sigma = more blur, use
                sigma ≈ kernel_size/6
              </li>
              <li>
                • <strong>Canny Thresholds:</strong> High threshold should be
                2-3× low threshold
              </li>
              <li>
                • <strong>Median Filter:</strong> Excellent for salt-and-pepper
                noise, preserves edges
              </li>
              <li>
                • <strong>Bilateral Filter:</strong> Slow but excellent for
                edge-preserving smoothing
              </li>
              <li>
                • <strong>FFT Shifting:</strong> Always use fftshift() to center
                zero frequency
              </li>
              <li>
                • <strong>Frequency Filtering:</strong> Low-pass smooths,
                high-pass enhances edges
              </li>
              <li>
                • <strong>Hybrid Images:</strong> Test with sigma_low=10-15,
                sigma_high=2-4
              </li>
              <li>
                • <strong>Border Handling:</strong> REFLECT_101 usually gives
                best results
              </li>
              <li>
                • <strong>Performance:</strong> Frequency domain filtering
                faster for large kernels
              </li>
            </ul>
          </div>

          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg">
            <h3 className="font-semibold text-red-900 mb-3">
              ⚠️ Common Mistakes
            </h3>
            <ul className="space-y-2 text-red-800 text-sm">
              <li>
                • Forgetting to normalize kernels (sum should equal 1 for
                smoothing)
              </li>
              <li>• Using even kernel sizes (causes centering issues)</li>
              <li>
                • Not handling negative values after edge detection (use
                absolute value)
              </li>
              <li>• Forgetting np.clip() after arithmetic operations</li>
              <li>• Not converting to uint8 before displaying with OpenCV</li>
              <li>• Mixing up fftshift() and ifftshift() order</li>
              <li>
                • Creating masks with wrong data type (use uint8 or float)
              </li>
              <li>• Not considering computational cost of bilateral filter</li>
              <li>
                • Choosing inappropriate border handling for specific
                applications
              </li>
              <li>• Over-sharpening images (creates halos and artifacts)</li>
            </ul>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-3">
              📊 Performance Considerations
            </h3>
            <div className="space-y-3 text-blue-800 text-sm">
              <p>
                <strong>Filter Speed Comparison (fastest to slowest):</strong>
              </p>
              <ol className="list-decimal list-inside space-y-1 ml-4">
                <li>Box filter (simple averaging) - O(n)</li>
                <li>Gaussian filter (separable) - O(n × k)</li>
                <li>Median filter - O(n × k × log k)</li>
                <li>Bilateral filter - O(n × k² × σ)</li>
              </ol>

              <p className="mt-3">
                <strong>When to use frequency domain:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li>Large kernel sizes (&gt; 15×15)</li>
                <li>Multiple filtering operations</li>
                <li>Image compression tasks</li>
                <li>Periodic noise removal</li>
              </ul>

              <p className="mt-3">
                <strong>Memory usage:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li>FFT requires ~3× image size in memory</li>
                <li>Bilateral filter: ~10× slower than Gaussian</li>
                <li>Use ROI (Region of Interest) for large images</li>
              </ul>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
