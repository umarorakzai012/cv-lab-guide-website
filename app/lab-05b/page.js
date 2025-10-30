// app/lab-05b/page.js
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { BookOpen, Code, CheckCircle, Waves } from "lucide-react";

export default function Lab05B() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Lab 05B: Advanced Techniques
        </h1>
        <p className="text-slate-600 text-lg">
          Wavelet transforms, boundary detection, Hough transforms, and SIFT
          feature extraction
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
          <ConceptCard title="Wavelet Transformation" icon={Waves}>
            <div className="space-y-4 text-slate-700">
              <p>
                Wavelet transformation is a mathematical technique used in
                signal processing, data analysis, and image processing. It
                analyzes a signal or image in terms of its frequency components
                at different scales, providing time-frequency analysis
                particularly useful for non-stationary signals.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  Why Wavelets?
                </p>
                <ul className="space-y-2 text-blue-800">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Multi-resolution analysis:</strong> Examine
                      signals at different scales
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Localized analysis:</strong> Both time and
                      frequency information
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Compression:</strong> Efficient for image
                      compression (JPEG2000)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Denoising:</strong> Separate signal from noise at
                      different scales
                    </span>
                  </li>
                </ul>
              </div>

              <Accordion type="single" collapsible className="w-full mt-4">
                <AccordionItem value="cwt">
                  <AccordionTrigger>
                    Continuous Wavelet Transform (CWT)
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Uses continuous wavelet functions dilated and translated
                        across the signal. The wavelet function ψ(t) is scaled
                        by different frequencies and shifted across time.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">Key Features:</p>
                        <ul className="space-y-1 ml-2">
                          <li>
                            • <strong>Wavelet families:</strong> Morlet, Mexican
                            Hat, etc.
                          </li>
                          <li>
                            • <strong>Applications:</strong> Seismic analysis,
                            time-frequency analysis
                          </li>
                          <li>
                            • <strong>Output:</strong> Continuous scalogram
                            showing energy distribution
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="dwt">
                  <AccordionTrigger>
                    Discrete Wavelet Transform (DWT)
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Uses discrete wavelet functions to transform discrete
                        signal data. Scales and shifts are done in discrete
                        steps.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">Key Features:</p>
                        <ul className="space-y-1 ml-2">
                          <li>
                            • <strong>Wavelet families:</strong> Daubechies,
                            Haar, Symlet, Coiflet
                          </li>
                          <li>
                            • <strong>Decomposition:</strong> Splits signal into
                            approximation + detail coefficients
                          </li>
                          <li>
                            • <strong>Applications:</strong> Image compression,
                            denoising, feature extraction
                          </li>
                          <li>
                            • <strong>Multi-level:</strong> Can decompose
                            recursively for different scales
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="2d">
                  <AccordionTrigger>
                    2D Wavelet Transform for Images
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Applies wavelet transform in both horizontal and
                        vertical directions, producing 4 subbands:
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <ul className="space-y-1 ml-2">
                          <li>
                            • <strong>LL (Approximation):</strong> Low-pass in
                            both directions - smooth version
                          </li>
                          <li>
                            • <strong>LH (Horizontal detail):</strong> Vertical
                            edges
                          </li>
                          <li>
                            • <strong>HL (Vertical detail):</strong> Horizontal
                            edges
                          </li>
                          <li>
                            • <strong>HH (Diagonal detail):</strong> Diagonal
                            features
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>

          <ConceptCard title="Boundary Detection">
            <div className="space-y-4 text-slate-700">
              <p>
                Boundary detection (edge detection) identifies boundaries of
                objects within an image. These boundaries represent significant
                changes in intensity or color and are essential for object
                recognition, segmentation, and feature extraction.
              </p>

              <div className="grid md:grid-cols-3 gap-4 mt-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Sobel Edge Detection
                  </h4>
                  <p className="text-sm text-green-800">
                    Basic gradient-based method using 3×3 kernels to calculate
                    horizontal and vertical gradients. Magnitude represents edge
                    strength.
                  </p>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    Canny Edge Detection
                  </h4>
                  <p className="text-sm text-blue-800">
                    Multi-stage: Gaussian smoothing → gradient calculation →
                    non-maximum suppression → hysteresis thresholding. High
                    accuracy, thin edges.
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Laplacian of Gaussian
                  </h4>
                  <p className="text-sm text-purple-800">
                    Applies Gaussian smoothing then Laplacian operator.
                    Zero-crossings indicate edges. Effective at varying scales.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Hough Transformation">
            <div className="space-y-4 text-slate-700">
              <p>
                The Hough Transform detects simple shapes (lines, circles,
                ellipses) in images. It transforms the spatial domain into a
                parameter space where shapes appear as points or curves.
                Particularly useful when traditional edge detection is
                insufficient.
              </p>

              <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
                <p className="font-semibold text-orange-900 mb-2">
                  Core Concept:
                </p>
                <p className="text-orange-800 text-sm">
                  Each point in image space corresponds to a curve/line in
                  parameter space. Multiple points on the same shape will have
                  intersecting curves in parameter space, creating "peaks" that
                  identify the shape.
                </p>
              </div>

              <Accordion type="single" collapsible className="w-full mt-4">
                <AccordionItem value="line">
                  <AccordionTrigger>
                    Hough Transform for Line Detection
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        <strong>Parametric Representation:</strong> Line in
                        Cartesian coordinates y = mx + b is represented in
                        parameter space with 'm' (slope) and 'b' (y-intercept)
                        as axes.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">Algorithm Steps:</p>
                        <ol className="list-decimal list-inside space-y-1 ml-2">
                          <li>
                            <strong>Edge Detection:</strong> Find edge pixels
                            using Canny or similar
                          </li>
                          <li>
                            <strong>Parameter Space:</strong> Use polar
                            coordinates ρ = x·cos(θ) + y·sin(θ)
                          </li>
                          <li>
                            <strong>Voting:</strong> Each edge pixel votes for
                            all possible lines through it
                          </li>
                          <li>
                            <strong>Peak Detection:</strong> Find peaks in
                            accumulator array
                          </li>
                          <li>
                            <strong>Back-Transform:</strong> Convert peaks back
                            to image space lines
                          </li>
                        </ol>
                      </div>
                      <p className="text-xs mt-2">
                        <strong>Note:</strong> Polar representation (ρ, θ)
                        preferred over (m, b) to handle vertical lines
                      </p>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="circle">
                  <AccordionTrigger>
                    Hough Transform for Circle Detection
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        <strong>Parametric Representation:</strong> Circle
                        equation (x - a)² + (y - b)² = r², where (a, b) is
                        center and r is radius.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">Algorithm Steps:</p>
                        <ol className="list-decimal list-inside space-y-1 ml-2">
                          <li>
                            <strong>Parameter Space:</strong> 3D space with axes
                            a, b, and r
                          </li>
                          <li>
                            <strong>Voting:</strong> Each edge pixel votes for
                            potential centers and radii
                          </li>
                          <li>
                            <strong>Peak Detection:</strong> Find peaks in 3D
                            accumulator
                          </li>
                          <li>
                            <strong>Back-Transform:</strong> Extract circle
                            parameters from peaks
                          </li>
                        </ol>
                      </div>
                      <p className="text-xs mt-2">
                        <strong>Challenge:</strong> 3D parameter space is
                        computationally expensive. Often use gradient direction
                        to reduce search space.
                      </p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="grid md:grid-cols-2 gap-4 mt-4">
                <div className="bg-green-50 p-3 rounded">
                  <h5 className="font-semibold text-green-900 text-sm mb-2">
                    ✓ Advantages
                  </h5>
                  <ul className="text-xs text-green-700 space-y-1">
                    <li>• Robust to noise and gaps in edges</li>
                    <li>• Can detect multiple shapes simultaneously</li>
                    <li>• Works with partial or occluded shapes</li>
                    <li>• Handles varying orientations well</li>
                  </ul>
                </div>
                <div className="bg-red-50 p-3 rounded">
                  <h5 className="font-semibold text-red-900 text-sm mb-2">
                    ✗ Limitations
                  </h5>
                  <ul className="text-xs text-red-700 space-y-1">
                    <li>• Computationally expensive (especially circles)</li>
                    <li>• Memory intensive for high-resolution images</li>
                    <li>• Requires good edge detection preprocessing</li>
                    <li>• Parameter tuning can be challenging</li>
                  </ul>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="SIFT Feature Extraction">
            <div className="space-y-4 text-slate-700">
              <p>
                SIFT (Scale-Invariant Feature Transform) is a powerful technique
                for feature extraction and matching. It's particularly useful
                for object recognition, image stitching, and tracking because
                features are invariant to scale, rotation, and partially
                invariant to illumination and viewpoint changes.
              </p>

              <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                <p className="font-semibold text-purple-900 mb-2">
                  SIFT Pipeline Overview:
                </p>
                <div className="text-purple-800 text-sm space-y-1">
                  <p>
                    1. <strong>Scale-Space Extrema Detection</strong> → Find
                    keypoint candidates
                  </p>
                  <p>
                    2. <strong>Keypoint Localization</strong> → Refine positions
                    and filter weak points
                  </p>
                  <p>
                    3. <strong>Orientation Assignment</strong> → Make
                    rotation-invariant
                  </p>
                  <p>
                    4. <strong>Descriptor Generation</strong> → Create
                    distinctive fingerprint
                  </p>
                  <p>
                    5. <strong>Keypoint Matching</strong> → Find correspondences
                    between images
                  </p>
                </div>
              </div>

              <Accordion type="single" collapsible className="w-full mt-4">
                <AccordionItem value="detection">
                  <AccordionTrigger>
                    Scale-Space Extrema Detection
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Detects keypoint candidates at multiple scales using
                        Difference of Gaussians (DoG).
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">Process:</p>
                        <ul className="space-y-1 ml-2">
                          <li>
                            • Create scale space by repeatedly blurring and
                            downsampling image
                          </li>
                          <li>
                            • Calculate DoG: Subtract two consecutive
                            Gaussian-blurred images
                          </li>
                          <li>
                            • Find local extrema by comparing each pixel to 26
                            neighbors (3×3×3 cube)
                          </li>
                          <li>• These extrema are keypoint candidates</li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="localization">
                  <AccordionTrigger>Keypoint Localization</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Refines keypoint locations and eliminates weak or
                        edge-like keypoints.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <ul className="space-y-1 ml-2">
                          <li>
                            • Fit 3D quadratic function to nearby data for
                            sub-pixel accuracy
                          </li>
                          <li>
                            • Reject low-contrast keypoints (threshold on DoG
                            value)
                          </li>
                          <li>
                            • Eliminate edge responses using Hessian matrix
                          </li>
                          <li>• Result: Stable, well-localized keypoints</li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="orientation">
                  <AccordionTrigger>Orientation Assignment</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Assigns consistent orientation to each keypoint based on
                        local gradient direction.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <ul className="space-y-1 ml-2">
                          <li>
                            • Calculate gradient magnitude and orientation in
                            neighborhood
                          </li>
                          <li>
                            • Create histogram of gradient orientations (36 bins
                            for 360°)
                          </li>
                          <li>• Peak in histogram = dominant orientation</li>
                          <li>
                            • Allows descriptor to be normalized relative to
                            this orientation
                          </li>
                          <li>
                            • Multiple orientations possible → multiple
                            keypoints at same location
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="descriptor">
                  <AccordionTrigger>
                    Keypoint Descriptor Generation
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Creates a 128-dimensional feature vector that uniquely
                        describes the local region around each keypoint.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">
                          Descriptor Construction:
                        </p>
                        <ul className="space-y-1 ml-2">
                          <li>
                            • 16×16 region around keypoint → divided into 4×4
                            sub-regions
                          </li>
                          <li>
                            • Each sub-region: 8-bin histogram of gradient
                            orientations
                          </li>
                          <li>• Total: 4×4×8 = 128 values</li>
                          <li>
                            • Normalized to unit length for illumination
                            invariance
                          </li>
                          <li>
                            • Values clamped to 0.2 and renormalized for further
                            robustness
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="matching">
                  <AccordionTrigger>Keypoint Matching</AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3 text-sm">
                      <p>
                        Match keypoints between images using descriptor
                        similarity.
                      </p>
                      <div className="bg-slate-100 p-3 rounded">
                        <p className="font-semibold mb-2">
                          Matching Strategies:
                        </p>
                        <ul className="space-y-1 ml-2">
                          <li>
                            • <strong>Euclidean Distance:</strong> Calculate L2
                            distance between descriptors
                          </li>
                          <li>
                            • <strong>Ratio Test (Lowe's):</strong> Distance to
                            nearest / distance to 2nd nearest &lt; 0.8
                          </li>
                          <li>
                            • <strong>RANSAC:</strong> Robust estimation to
                            reject outliers
                          </li>
                          <li>
                            • <strong>Homography Estimation:</strong> Find
                            geometric transformation
                          </li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="bg-yellow-50 p-4 rounded-lg mt-4">
                <p className="font-semibold text-yellow-900 mb-2 text-sm">
                  🎯 SIFT Applications:
                </p>
                <div className="grid md:grid-cols-2 gap-2 text-xs text-yellow-800">
                  <div>
                    <p>
                      • <strong>Object Recognition:</strong> Identify objects in
                      cluttered scenes
                    </p>
                    <p>
                      • <strong>Image Stitching:</strong> Create panoramas from
                      multiple images
                    </p>
                    <p>
                      • <strong>3D Reconstruction:</strong> Structure from
                      motion
                    </p>
                  </div>
                  <div>
                    <p>
                      • <strong>Robot Navigation:</strong> Visual SLAM and
                      localization
                    </p>
                    <p>
                      • <strong>Tracking:</strong> Follow objects across video
                      frames
                    </p>
                    <p>
                      • <strong>Image Registration:</strong> Align medical
                      images
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="2D Discrete Wavelet Transform">
            <div className="space-y-4">
              <p className="text-slate-700">
                Apply 2D DWT to decompose an image into approximation and detail
                coefficients at multiple scales.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply 2D Discrete Wavelet Transform
# wavelet: 'haar', 'db1', 'db2', 'sym2', 'coif1', etc.
wavelet = 'haar'
coeffs = pywt.dwt2(image, wavelet)

# Decompose into approximation and details
cA, (cH, cV, cD) = coeffs
# cA: Approximation (low-pass in both directions)
# cH: Horizontal detail (vertical edges)
# cV: Vertical detail (horizontal edges)
# cD: Diagonal detail

# Multi-level decomposition
levels = 3
coeffs_multilevel = pywt.wavedec2(image, wavelet, level=levels)

# Reconstruct image from coefficients
reconstructed = pywt.waverec2(coeffs_multilevel, wavelet)

# Normalize coefficients for visualization
def normalize_coeff(coeff):
    return np.uint8(255 * (coeff - coeff.min()) / (coeff.max() - coeff.min()))

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(normalize_coeff(cA), cmap='gray')
axes[0, 1].set_title('Approximation (cA)')
axes[0, 1].axis('off')

axes[0, 2].imshow(normalize_coeff(cH), cmap='gray')
axes[0, 2].set_title('Horizontal Detail (cH)')
axes[0, 2].axis('off')

axes[1, 0].imshow(normalize_coeff(cV), cmap='gray')
axes[1, 0].set_title('Vertical Detail (cV)')
axes[1, 0].axis('off')

axes[1, 1].imshow(normalize_coeff(cD), cmap='gray')
axes[1, 1].set_title('Diagonal Detail (cD)')
axes[1, 1].axis('off')

axes[1, 2].imshow(reconstructed, cmap='gray')
axes[1, 2].set_title('Reconstructed')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Multi-level visualization
fig, axes = plt.subplots(1, levels + 1, figsize=(16, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i in range(levels):
    # Reconstruct from each level
    coeffs_partial = coeffs_multilevel[:i+2] + [None] * (levels - i - 1)
    recon = pywt.waverec2(coeffs_partial, wavelet)
    axes[i+1].imshow(recon, cmap='gray')
    axes[i+1].set_title(f'Level {i+1}')
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()

print(f"Original image shape: {image.shape}")
print(f"Approximation shape: {cA.shape}")
print(f"Detail coefficient shapes: {cH.shape}")
print(f"\nWavelet family: {wavelet}")
print(f"Decomposition levels: {levels}")
print(f"Compression ratio: {image.size / cA.size:.2f}x")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Wavelet Denoising">
            <div className="space-y-4">
              <p className="text-slate-700">
                Use wavelet transform for image denoising by thresholding detail
                coefficients.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise_sigma = 25
noisy_image = image + noise_sigma * np.random.randn(*image.shape)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Wavelet denoising function
def wavelet_denoise(img, wavelet='db1', level=2, threshold_type='soft'):
    """
    Denoise image using wavelet transform
    
    Parameters:
    - img: input image
    - wavelet: wavelet family
    - level: decomposition level
    - threshold_type: 'soft' or 'hard'
    """
    # Decompose
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    
    # Estimate noise standard deviation from finest scale
    sigma_est = np.median(np.abs(coeffs[-1][2])) / 0.6745
    
    # Calculate threshold (VisuShrink)
    threshold = sigma_est * np.sqrt(2 * np.log(img.size))
    
    # Apply threshold to detail coefficients
    new_coeffs = [coeffs[0]]  # Keep approximation
    for detail_level in coeffs[1:]:
        new_detail = list(detail_level)
        for i in range(3):  # cH, cV, cD
            if threshold_type == 'soft':
                new_detail[i] = pywt.threshold(detail_level[i], threshold, mode='soft')
            else:
                new_detail[i] = pywt.threshold(detail_level[i], threshold, mode='hard')
        new_coeffs.append(tuple(new_detail))
    
    # Reconstruct
    denoised = pywt.waverec2(new_coeffs, wavelet)
    return np.clip(denoised, 0, 255).astype(np.uint8)

# Try different wavelet families and thresholding
wavelets = ['haar', 'db4', 'sym4']
results = []

for wavelet in wavelets:
    denoised_soft = wavelet_denoise(noisy_image, wavelet=wavelet, 
                                    level=2, threshold_type='soft')
    denoised_hard = wavelet_denoise(noisy_image, wavelet=wavelet,
                                    level=2, threshold_type='hard')
    results.append((wavelet, denoised_soft, denoised_hard))

# Calculate PSNR
def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Display results
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_image, cmap='gray')
axes[0, 1].set_title(f'Noisy (σ={noise_sigma})')
axes[0, 1].axis('off')

# Compare with Gaussian blur
gaussian_denoised = cv2.GaussianBlur(noisy_image, (5, 5), 0)
axes[0, 2].imshow(gaussian_denoised, cmap='gray')
axes[0, 2].set_title('Gaussian Blur')
axes[0, 2].axis('off')

for idx, (wavelet, soft, hard) in enumerate(results):
    row = idx + 1
    
    psnr_soft = calculate_psnr(image, soft)
    psnr_hard = calculate_psnr(image, hard)
    
    axes[row, 0].imshow(soft, cmap='gray')
    axes[row, 0].set_title(f'{wavelet} - Soft\\nPSNR: {psnr_soft:.2f}dB')
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(hard, cmap='gray')
    axes[row, 1].set_title(f'{wavelet} - Hard\\nPSNR: {psnr_hard:.2f}dB')
    axes[row, 1].axis('off')
    
    # Show difference
    diff = np.abs(image.astype(float) - soft.astype(float))
    axes[row, 2].imshow(diff, cmap='hot')
    axes[row, 2].set_title('Error Map (Soft)')
    axes[row, 2].axis('off')

plt.tight_layout()
plt.show()

# Print analysis
print("Wavelet Denoising Analysis:")
print("=" * 60)
psnr_gaussian = calculate_psnr(image, gaussian_denoised)
print(f"Gaussian Blur PSNR: {psnr_gaussian:.2f} dB")
print()
for wavelet, soft, hard in results:
    psnr_soft = calculate_psnr(image, soft)
    psnr_hard = calculate_psnr(image, hard)
    print(f"{wavelet} wavelet:")
    print(f"  Soft thresholding PSNR: {psnr_soft:.2f} dB")
    print(f"  Hard thresholding PSNR: {psnr_hard:.2f} dB")
    print()

print("=" * 60)
print("Key Observations:")
print("- Wavelet denoising preserves edges better than Gaussian blur")
print("- Soft thresholding typically gives smoother results")
print("- Hard thresholding preserves more detail but may have artifacts")
print("- Different wavelets work better for different image types")`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Hough Line Transform">
            <div className="space-y-4">
              <p className="text-slate-700">
                Detect straight lines in images using Hough Transform, useful
                for detecting lane markings, document edges, and architectural
                features.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Step 2: Standard Hough Line Transform
# Returns lines in (rho, theta) format
lines_standard = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Draw standard Hough lines
img_hough_standard = image_rgb.copy()
if lines_standard is not None:
    for line in lines_standard:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend line to image boundaries
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_hough_standard, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Step 3: Probabilistic Hough Line Transform
# Returns lines as (x1, y1, x2, y2) - more efficient
lines_prob = cv2.HoughLinesP(
    edges, 
    rho=1,              # Distance resolution in pixels
    theta=np.pi/180,    # Angle resolution in radians
    threshold=50,       # Min number of votes
    minLineLength=50,   # Min line length
    maxLineGap=10       # Max gap between segments
)

# Draw probabilistic Hough lines
img_hough_prob = image_rgb.copy()
if lines_prob is not None:
    for line in lines_prob:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_hough_prob, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Try different parameters
param_configs = [
    {'threshold': 100, 'minLineLength': 50, 'maxLineGap': 10},
    {'threshold': 50, 'minLineLength': 30, 'maxLineGap': 5},
    {'threshold': 150, 'minLineLength': 100, 'maxLineGap': 20}
]

results_params = []
for config in param_configs:
    lines_temp = cv2.HoughLinesP(edges, 1, np.pi/180, **config)
    img_temp = image_rgb.copy()
    if lines_temp is not None:
        for line in lines_temp:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_temp, (x1, y1), (x2, y2), (255, 0, 255), 2)
    results_params.append((config, img_temp, len(lines_temp) if lines_temp is not None else 0))

# Display results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(edges, cmap='gray')
axes[0, 1].set_title('Canny Edges')
axes[0, 1].axis('off')

axes[0, 2].imshow(img_hough_standard)
axes[0, 2].set_title(f'Standard Hough\\n{len(lines_standard) if lines_standard is not None else 0} lines')
axes[0, 2].axis('off')

axes[1, 0].imshow(img_hough_prob)
axes[1, 0].set_title(f'Probabilistic Hough\\n{len(lines_prob) if lines_prob is not None else 0} lines')
axes[1, 0].axis('off')

for idx, (config, img, count) in enumerate(results_params[:2]):
    axes[1, idx+1].imshow(img)
    title = f"T={config['threshold']}, Min={config['minLineLength']}\\n{count} lines"
    axes[1, idx+1].set_title(title)
    axes[1, idx+1].axis('off')

plt.tight_layout()
plt.show()

# Analysis
print("Hough Line Transform Analysis:")
print("=" * 60)
print(f"Standard Hough Lines detected: {len(lines_standard) if lines_standard is not None else 0}")
print(f"Probabilistic Hough Lines detected: {len(lines_prob) if lines_prob is not None else 0}")
print()

print("Parameter Effects:")
for config, _, count in results_params:
    print(f"  Threshold={config['threshold']}, MinLength={config['minLineLength']}, MaxGap={config['maxLineGap']}")
    print(f"    Lines detected: {count}")

print()
print("=" * 60)
print("Tips:")
print("- Higher threshold: Fewer but more confident lines")
print("- Lower minLineLength: More line segments detected")
print("- Higher maxLineGap: Connects more fragmented lines")
print("- Probabilistic version is faster and returns line segments")
print("=" * 60)`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Hough Circle Transform">
            <div className="space-y-4">
              <p className="text-slate-700">
                Detect circles in images using Hough Circle Transform, perfect
                for coin detection, circular object recognition, and iris
                detection.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('circles.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise (important for circle detection)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Hough Circle Transform
# cv2.HoughCircles returns circles as (x, y, radius)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,     # Detection method
    dp=1,                    # Inverse ratio of accumulator resolution
    minDist=50,              # Minimum distance between circle centers
    param1=100,              # Upper threshold for Canny edge detector
    param2=30,               # Accumulator threshold (lower = more circles)
    minRadius=10,            # Minimum circle radius
    maxRadius=100            # Maximum circle radius
)

# Draw detected circles
img_circles = image_rgb.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, (x, y, r) in enumerate(circles[0, :]):
        # Draw outer circle
        cv2.circle(img_circles, (x, y), r, (0, 255, 0), 3)
        # Draw center
        cv2.circle(img_circles, (x, y), 2, (255, 0, 0), 3)
        # Add label
        cv2.putText(img_circles, f'{i+1}', (x-10, y-r-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Try different parameter combinations
param_configs = [
    {'param2': 20, 'minRadius': 10, 'maxRadius': 100},  # More sensitive
    {'param2': 30, 'minRadius': 10, 'maxRadius': 100},  # Balanced
    {'param2': 50, 'minRadius': 10, 'maxRadius': 100},  # More strict
    {'param2': 30, 'minRadius': 20, 'maxRadius': 80},   # Size constraint
]

results_params = []
for config in param_configs:
    circles_temp = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, **config
    )
    
    img_temp = image_rgb.copy()
    count = 0
    if circles_temp is not None:
        circles_temp = np.uint16(np.around(circles_temp))
        count = len(circles_temp[0])
        for x, y, r in circles_temp[0, :]:
            cv2.circle(img_temp, (x, y), r, (255, 0, 255), 2)
            cv2.circle(img_temp, (x, y), 2, (0, 255, 255), 3)
    
    results_params.append((config, img_temp, count))

# Display results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Grayscale')
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('Blurred')
axes[0, 2].axis('off')

axes[1, 0].imshow(img_circles)
axes[1, 0].set_title(f'Detected Circles\\n{len(circles[0]) if circles is not None else 0} circles')
axes[1, 0].axis('off')

for idx, (config, img, count) in enumerate(results_params[:2]):
    axes[1, idx+1].imshow(img)
    title = f"param2={config['param2']}, r=[{config['minRadius']},{config['maxRadius']}]\\n{count} circles"
    axes[1, idx+1].set_title(title, fontsize=9)
    axes[1, idx+1].axis('off')

plt.tight_layout()
plt.show()

# Detailed analysis
print("Circle Detection Analysis:")
print("=" * 60)

if circles is not None:
    print(f"Total circles detected: {len(circles[0])}")
    print()
    print("Circle Details:")
    for i, (x, y, r) in enumerate(circles[0, :]):
        area = np.pi * r * r
        print(f"  Circle {i+1}: Center=({x}, {y}), Radius={r}, Area={area:.1f} px²")
else:
    print("No circles detected!")

print()
print("Parameter Configuration Results:")
for config, _, count in results_params:
    print(f"  param2={config['param2']}, radius=[{config['minRadius']},{config['maxRadius']}]: {count} circles")

print()
print("=" * 60)
print("Parameter Guide:")
print("- dp: 1 = same resolution as input, 2 = half resolution")
print("- minDist: Minimum distance between detected circle centers")
print("- param1: Upper threshold for Canny edge detection")
print("- param2: Accumulator threshold (LOWER = more circles)")
print("- minRadius/maxRadius: Constrain circle size range")
print()
print("Tips:")
print("- Blur image first to reduce noise")
print("- Start with higher param2, decrease to find more circles")
print("- Adjust minDist to prevent overlapping detections")
print("- Use radius constraints if you know approximate sizes")
print("=" * 60)`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="SIFT Feature Detection and Matching">
            <div className="space-y-4">
              <p className="text-slate-700">
                Extract SIFT features from images and match them between
                different views for object recognition and image alignment.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images (same object from different viewpoints)
img1 = cv2.imread('object1.jpg')
img2 = cv2.imread('object2.jpg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create(
    nfeatures=0,           # Keep all features (0 = unlimited)
    nOctaveLayers=3,       # Number of layers in each octave
    contrastThreshold=0.04, # Threshold to filter weak features
    edgeThreshold=10,      # Threshold to filter edge-like features
    sigma=1.6              # Gaussian sigma for initial image
)

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

print(f"Image 1: {len(keypoints1)} keypoints")
print(f"Image 2: {len(keypoints2)} keypoints")
print(f"Descriptor shape: {descriptors1.shape}")

# Draw keypoints
img1_kp = cv2.drawKeypoints(img1_rgb, keypoints1, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2_rgb, keypoints2, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# KNN matching (k=2 for ratio test)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test (Lowe's ratio test)
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"\\nMatches found: {len(matches)}")
print(f"Good matches (after ratio test): {len(good_matches)}")

# Sort matches by distance
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Draw matches
img_matches = cv2.drawMatches(img1_rgb, keypoints1, img2_rgb, keypoints2,
                               good_matches[:50], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Find homography using RANSAC
if len(good_matches) >= 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    # Count inliers
    inliers = np.sum(matchesMask)
    print(f"Inliers (RANSAC): {inliers}/{len(good_matches)}")
    
    # Draw only inlier matches
    inlier_matches = [m for m, keep in zip(good_matches, matchesMask) if keep]
    img_inliers = cv2.drawMatches(img1_rgb, keypoints1, img2_rgb, keypoints2,
                                   inlier_matches[:50], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Warp image 1 to align with image 2
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    img1_aligned = cv2.warpPerspective(img1_rgb, M, (w2, h2))
else:
    img_inliers = img_matches
    img1_aligned = img1_rgb
    print("Not enough matches for homography")

# Display results
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)

# Row 1: Original images with keypoints
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img1_kp)
ax1.set_title(f'Image 1 - {len(keypoints1)} Keypoints')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img2_kp)
ax2.set_title(f'Image 2 - {len(keypoints2)} Keypoints')
ax2.axis('off')

# Row 2: All matches
ax3 = fig.add_subplot(gs[1, :])
ax3.imshow(img_matches)
ax3.set_title(f'Top 50 Matches (of {len(good_matches)} good matches)')
ax3.axis('off')

# Row 3: Inlier matches only
ax4 = fig.add_subplot(gs[2, :])
ax4.imshow(img_inliers)
ax4.set_title(f'Inlier Matches after RANSAC')
ax4.axis('off')

# Row 4: Alignment result
ax5 = fig.add_subplot(gs[3, 0])
ax5.imshow(img1_aligned)
ax5.set_title('Image 1 Aligned to Image 2')
ax5.axis('off')

ax6 = fig.add_subplot(gs[3, 1])
ax6.imshow(img2_rgb)
ax6.set_title('Image 2 (Reference)')
ax6.axis('off')

plt.suptitle('SIFT Feature Detection and Matching', fontsize=16, y=0.98)
plt.show()

# Analyze keypoint properties
print("\\n" + "=" * 60)
print("SIFT Keypoint Analysis:")
print("=" * 60)

# Keypoint sizes (scales)
sizes1 = [kp.size for kp in keypoints1]
sizes2 = [kp.size for kp in keypoints2]

print(f"\\nKeypoint Scales (Image 1):")
print(f"  Mean: {np.mean(sizes1):.2f}")
print(f"  Min: {np.min(sizes1):.2f}, Max: {np.max(sizes1):.2f}")

print(f"\\nKeypoint Scales (Image 2):")
print(f"  Mean: {np.mean(sizes2):.2f}")
print(f"  Min: {np.min(sizes2):.2f}, Max: {np.max(sizes2):.2f}")

# Match distances
if len(good_matches) > 0:
    distances = [m.distance for m in good_matches[:50]]
    print(f"\\nTop 50 Match Distances:")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Min: {np.min(distances):.2f}, Max: {np.max(distances):.2f}")

print("\\n" + "=" * 60)
print("SIFT Properties:")
print("=" * 60)
print("✓ Scale Invariant: Works across different image scales")
print("✓ Rotation Invariant: Keypoints have consistent orientation")
print("✓ Illumination Robust: Partially invariant to lighting changes")
print("✓ Viewpoint Robust: Handles moderate viewpoint changes")
print("✓ 128D Descriptor: Distinctive and discriminative")
print("=" * 60)`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Panorama Stitching with SIFT">
            <div className="space-y-4">
              <p className="text-slate-700">
                Create panoramic images by stitching multiple overlapping images
                using SIFT features for alignment.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(img1, img2):
    """Stitch two images together using SIFT"""
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    print(f"  Found {len(good)} good matches")
    
    # Find homography
    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of img1
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transform corners of img1
        pts1_transformed = cv2.perspectiveTransform(pts1, M)
        pts = np.concatenate((pts1_transformed, pts2), axis=0)
        
        # Get bounding box
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        
        # Translation matrix
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        
        # Warp img1
        result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
        
        # Place img2
        result[t[1]:t[1]+h2, t[0]:t[0]+w2] = img2
        
        return result, len(good), mask.ravel().sum()
    else:
        print("  Not enough matches!")
        return None, len(good), 0

# Load images for panorama (should overlap)
images = []
image_files = ['pano1.jpg', 'pano2.jpg', 'pano3.jpg']

print("Loading images...")
for file in image_files:
    img = cv2.imread(file)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        print(f"  {file}: {img.shape}")

if len(images) < 2:
    print("Need at least 2 images for stitching!")
else:
    print(f"\\nLoaded {len(images)} images")
    
    # Stitch images sequentially
    print("\\nStitching images...")
    panorama = images[0]
    
    stats = []
    for i in range(1, len(images)):
        print(f"Stitching image {i+1}...")
        result, matches, inliers = stitch_images(panorama, images[i])
        if result is not None:
            panorama = result
            stats.append((i, matches, inliers))
            print(f"  Success! Inliers: {inliers}/{matches}")
        else:
            print(f"  Failed to stitch image {i+1}")
            break
    
    # Crop black borders (optional)
    gray_pano = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_pano, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        panorama_cropped = panorama[y:y+h, x:x+w]
    else:
        panorama_cropped = panorama
    
    # Display results
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(len(images) + 2, 1, hspace=0.3)
    
    # Show original images
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    # Show final panorama
    ax1 = fig.add_subplot(gs[len(images), 0])
    ax1.imshow(panorama)
    ax1.set_title('Panorama (with borders)')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[len(images)+1, 0])
    ax2.imshow(panorama_cropped)
    ax2.set_title('Panorama (cropped)')
    ax2.axis('off')
    
    plt.suptitle('Panoramic Image Stitching with SIFT', fontsize=16, y=0.98)
    plt.show()
    
    # Print statistics
    print("\\n" + "=" * 60)
    print("Stitching Statistics:")
    print("=" * 60)
    for idx, matches, inliers in stats:
        print(f"Image {idx} -> {idx+1}:")
        print(f"  Matches: {matches}")
        print(f"  Inliers: {inliers} ({inliers/matches*100:.1f}%)")
    
    print(f"\\nFinal panorama size: {panorama_cropped.shape}")
    print("=" * 60)`}
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
              Apply advanced computer vision techniques to solve real-world
              problems.
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="task1">
              <AccordionTrigger className="text-lg font-semibold">
                Task 1: Computer Screen Detection using Hough Lines
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="font-semibold text-blue-900 mb-2">Scenario:</p>
                  <p className="text-blue-800 text-sm">
                    You're monitoring a computer lab to track screen
                    availability. Implement screen detection using Hough Line
                    Transformation to identify screen boundaries and detect
                    anomalies like missing screens.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load computer lab image
lab_image = cv2.imread('computer_lab.jpg')
lab_rgb = cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(lab_image, cv2.COLOR_BGR2GRAY)

# Preprocessing
# Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Reduce noise
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Edge detection with Canny
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Apply morphological operations to connect edges
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(
    edges_closed,
    rho=1,
    theta=np.pi/180,
    threshold=80,
    minLineLength=100,
    maxLineGap=10
)

# Separate horizontal and vertical lines
horizontal_lines = []
vertical_lines = []
angle_threshold = 15  # degrees

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        if x2 - x1 != 0:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        else:
            angle = 90
        
        # Classify as horizontal or vertical
        if angle < angle_threshold:
            horizontal_lines.append((x1, y1, x2, y2))
        elif angle > (90 - angle_threshold):
            vertical_lines.append((x1, y1, x2, y2))

print(f"Detected {len(horizontal_lines)} horizontal lines")
print(f"Detected {len(vertical_lines)} vertical lines")

# Function to find screen rectangles
def find_screens(h_lines, v_lines, img_shape):
    """Find rectangular screens from line intersections"""
    screens = []
    h, w = img_shape[:2]
    
    # Cluster lines by position
    def cluster_lines(lines, is_horizontal):
        if not lines:
            return []
        
        clustered = []
        lines = sorted(lines, key=lambda l: l[1] if is_horizontal else l[0])
        
        current_cluster = [lines[0]]
        threshold = 20  # pixels
        
        for line in lines[1:]:
            ref_coord = line[1] if is_horizontal else line[0]
            cluster_ref = current_cluster[-1][1] if is_horizontal else current_cluster[-1][0]
            
            if abs(ref_coord - cluster_ref) < threshold:
                current_cluster.append(line)
            else:
                clustered.append(current_cluster)
                current_cluster = [line]
        
        clustered.append(current_cluster)
        return clustered
    
    h_clusters = cluster_lines(h_lines, True)
    v_clusters = cluster_lines(v_lines, False)
    
    print(f"\\nClustered into {len(h_clusters)} horizontal and {len(v_clusters)} vertical groups")
    
    # Find rectangles from line intersections
    for i, h_cluster1 in enumerate(h_clusters):
        for h_cluster2 in h_clusters[i+1:]:
            for v_cluster1 in v_clusters:
                for v_cluster2 in v_clusters:
                    # Get representative positions
                    y1 = int(np.mean([l[1] for l in h_cluster1]))
                    y2 = int(np.mean([l[1] for l in h_cluster2]))
                    x1 = int(np.mean([l[0] for l in v_cluster1]))
                    x2 = int(np.mean([l[0] for l in v_cluster2]))
                    
                    # Validate rectangle
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    # Screen aspect ratio approximately 16:9 or 4:3
                    if 50 < width < w//2 and 50 < height < h//2:
                        aspect = width / height
                        if 1.2 < aspect < 2.0:  # Valid screen aspect ratios
                            screens.append((min(x1,x2), min(y1,y2), 
                                          max(x1,x2), max(y1,y2)))
    
    return screens

# Find screen rectangles
screens = find_screens(horizontal_lines, vertical_lines, lab_rgb.shape)

# Remove overlapping detections (non-maximum suppression)
def non_max_suppression(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(areas)
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick].astype(int)

screens_filtered = non_max_suppression(screens)

print(f"\\nDetected {len(screens_filtered)} computer screens")

# Draw results
img_lines = lab_rgb.copy()
img_screens = lab_rgb.copy()

# Draw all lines
for x1, y1, x2, y2 in horizontal_lines:
    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
for x1, y1, x2, y2 in vertical_lines:
    cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Draw detected screens
for idx, (x1, y1, x2, y2) in enumerate(screens_filtered):
    cv2.rectangle(img_screens, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Calculate screen status (simple brightness check)
    screen_roi = gray[y1:y2, x1:x2]
    mean_brightness = np.mean(screen_roi)
    
    status = "ON" if mean_brightness > 50 else "OFF"
    color = (0, 255, 0) if status == "ON" else (255, 0, 0)
    
    cv2.putText(img_screens, f"Screen {idx+1}: {status}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 2)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(lab_rgb)
axes[0, 0].set_title('Original Lab Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(enhanced, cmap='gray')
axes[0, 1].set_title('Enhanced (CLAHE)')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges, cmap='gray')
axes[0, 2].set_title('Canny Edges')
axes[0, 2].axis('off')

axes[1, 0].imshow(edges_closed, cmap='gray')
axes[1, 0].set_title('Morphological Closing')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_lines)
axes[1, 1].set_title(f'Detected Lines\\nH:{len(horizontal_lines)} V:{len(vertical_lines)}')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_screens)
axes[1, 2].set_title(f'Detected Screens: {len(screens_filtered)}')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Analysis report
print("\\n" + "="*60)
print("COMPUTER SCREEN DETECTION REPORT")
print("="*60)

for idx, (x1, y1, x2, y2) in enumerate(screens_filtered):
    width = x2 - x1
    height = y2 - y1
    aspect = width / height
    
    screen_roi = gray[y1:y2, x1:x2]
    mean_brightness = np.mean(screen_roi)
    status = "ON" if mean_brightness > 50 else "OFF"
    
    print(f"\\nScreen {idx+1}:")
    print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Aspect Ratio: {aspect:.2f}")
    print(f"  Status: {status} (brightness: {mean_brightness:.1f})")

print("\\n" + "="*60)
print("ANOMALIES DETECTED:")
print("="*60)

# Check for missing screens (if known layout)
expected_screens = 12  # example
if len(screens_filtered) < expected_screens:
    print(f"⚠ Warning: Only {len(screens_filtered)} screens detected")
    print(f"  Expected: {expected_screens}")
    print(f"  Missing: {expected_screens - len(screens_filtered)}")
else:
    print("✓ All expected screens detected")

print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task2">
              <AccordionTrigger className="text-lg font-semibold">
                Task 2: Asset Tracking Using SIFT
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="font-semibold text-purple-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-purple-800 text-sm">
                    Manage computer assets in a busy lab by implementing
                    SIFT-based recognition to automatically identify computers
                    and their components (monitors, keyboards) for inventory
                    maintenance.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Asset database - reference images of equipment
asset_database = {
    'monitor_dell': cv2.imread('monitor_dell.jpg'),
    'monitor_hp': cv2.imread('monitor_hp.jpg'),
    'keyboard_logitech': cv2.imread('keyboard_logitech.jpg'),
    'keyboard_dell': cv2.imread('keyboard_dell.jpg')
}

# Convert to RGB
for key in asset_database:
    if asset_database[key] is not None:
        asset_database[key] = cv2.cvtColor(asset_database[key], cv2.COLOR_BGR2RGB)

print("Asset Database loaded:")
for asset, img in asset_database.items():
    if img is not None:
        print(f"  {asset}: {img.shape}")

# Load current lab image
lab_scene = cv2.imread('lab_scene.jpg')
lab_scene_rgb = cv2.cvtColor(lab_scene, cv2.COLOR_BGR2RGB)

# Initialize SIFT
sift = cv2.SIFT_create(nfeatures=1000)

# Extract features for all assets
asset_features = {}
for asset_name, asset_img in asset_database.items():
    if asset_img is not None:
        gray = cv2.cvtColor(asset_img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        asset_features[asset_name] = {
            'keypoints': kp,
            'descriptors': des,
            'image': asset_img
        }
        print(f"  {asset_name}: {len(kp)} features")

# Extract features from lab scene
gray_scene = cv2.cvtColor(lab_scene_rgb, cv2.COLOR_RGB2GRAY)
kp_scene, des_scene = sift.detectAndCompute(gray_scene, None)
print(f"\\nLab scene: {len(kp_scene)} features")

# Function to detect asset in scene
def detect_asset(asset_name, asset_data, scene_kp, scene_des, scene_img):
    """Detect and localize asset in scene"""
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(asset_data['descriptors'], scene_des, k=2)
    
    # Apply ratio test
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good.append(m)
    
    print(f"\\n{asset_name}: {len(good)} good matches")
    
    # Need at least 10 matches for reliable detection
    if len(good) >= 10:
        # Get matched keypoints
        src_pts = np.float32([asset_data['keypoints'][m.queryIdx].pt 
                              for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kp[m.trainIdx].pt 
                              for m in good]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            inliers = mask.ravel().sum()
            print(f"  Inliers: {inliers}/{len(good)}")
            
            # Only accept if enough inliers
            if inliers >= 10:
                # Get corners of asset image
                h, w = asset_data['image'].shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                
                # Transform corners to scene
                dst = cv2.perspectiveTransform(pts, M)
                
                # Calculate confidence score
                confidence = (inliers / len(good)) * 100
                
                return {
                    'detected': True,
                    'corners': dst,
                    'matches': len(good),
                    'inliers': inliers,
                    'confidence': confidence,
                    'homography': M
                }
    
    return {'detected': False}

# Detect all assets in scene
detections = {}
for asset_name, asset_data in asset_features.items():
    result = detect_asset(asset_name, asset_data, kp_scene, 
                         des_scene, lab_scene_rgb)
    detections[asset_name] = result

# Visualize detections
img_detections = lab_scene_rgb.copy()
inventory = []

for asset_name, detection in detections.items():
    if detection['detected']:
        # Draw bounding box
        corners = np.int32(detection['corners'])
        cv2.polylines(img_detections, [corners], True, (0, 255, 0), 3)
        
        # Add label
        x, y = corners[0][0]
        label = f"{asset_name.replace('_', ' ').title()}"
        conf = f"{detection['confidence']:.1f}%"
        
        cv2.putText(img_detections, label, (x, y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_detections, conf, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add to inventory
        inventory.append({
            'asset': asset_name,
            'confidence': detection['confidence'],
            'location': corners.tolist(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Create detailed visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Show lab scene
ax1 = fig.add_subplot(gs[0, :])
ax1.imshow(img_detections)
ax1.set_title(f'Lab Scene - {len(inventory)} Assets Detected')
ax1.axis('off')

# Show individual asset matches
asset_items = list(asset_features.items())
for idx, (asset_name, asset_data) in enumerate(asset_items[:6]):
    row = 1 + idx // 3
    col = idx % 3
    
    if idx < 6:
        ax = fig.add_subplot(gs[row, col])
        
        detection = detections[asset_name]
        if detection['detected']:
            # Draw matches
            matches_to_draw = 20
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(asset_data['descriptors'], des_scene, k=2)
            good = [m for pair in matches if len(pair) == 2 
                   for m in [pair[0]] if m.distance < 0.7 * pair[1].distance]
            
            match_img = cv2.drawMatches(
                asset_data['image'], asset_data['keypoints'],
                lab_scene_rgb, kp_scene,
                good[:matches_to_draw], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            ax.imshow(match_img)
            ax.set_title(f"{asset_name}\\n✓ Detected ({detection['confidence']:.1f}%)")
        else:
            ax.imshow(asset_data['image'])
            ax.set_title(f"{asset_name}\\n✗ Not Detected")
        
        ax.axis('off')

plt.suptitle('Asset Tracking System - SIFT-based Recognition', 
             fontsize=16, y=0.98)
plt.show()

# Generate inventory report
print("\\n" + "="*70)
print("ASSET INVENTORY REPORT")
print("="*70)
print(f"Scan Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Location: Computer Lab A")
print("-"*70)

if inventory:
    print(f"\\nTotal Assets Detected: {len(inventory)}")
    print()
    
    for idx, item in enumerate(inventory, 1):
        print(f"{idx}. {item['asset'].replace('_', ' ').title()}")
        print(f"   Confidence: {item['confidence']:.2f}%")
        print(f"   Detection Time: {item['timestamp']}")
        
        # Calculate approximate position
        corners = np.array(item['location'])
        center_x = int(np.mean(corners[:, 0, 0]))
        center_y = int(np.mean(corners[:, 0, 1]))
        print(f"   Approx. Position: ({center_x}, {center_y})")
        print()
else:
    print("\\n⚠ No assets detected in current scan")

print("-"*70)

# Asset summary by category
monitors = [a for a in inventory if 'monitor' in a['asset']]
keyboards = [a for a in inventory if 'keyboard' in a['asset']]

print("\\nSUMMARY BY CATEGORY:")
print(f"  Monitors: {len(monitors)}")
print(f"  Keyboards: {len(keyboards)}")

# Check for missing assets
print("\\n" + "="*70)
print("ASSET STATUS:")
print("="*70)

for asset_name in asset_database.keys():
    detected = any(a['asset'] == asset_name for a in inventory)
    status = "✓ PRESENT" if detected else "✗ MISSING"
    print(f"  {asset_name.replace('_', ' ').title()}: {status}")

print("="*70)

# Recommendations
print("\\nRECOMMENDATIONS:")
if len(inventory) < len(asset_database):
    print("  • Some registered assets were not detected")
    print("  • Verify physical presence of missing items")
    print("  • Update asset database if items have been relocated")
else:
    print("  • All registered assets successfully detected")
    print("  • Inventory is up to date")

print("\\n" + "="*70)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task3">
              <AccordionTrigger className="text-lg font-semibold">
                Task 3: Anomaly Detection in Sensor Data Using Wavelets
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-orange-50 p-4 rounded-lg">
                  <p className="font-semibold text-orange-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-orange-800 text-sm">
                    Monitor sensor data from industrial machines and identify
                    anomalies in real-time. Use wavelet transformation to
                    analyze data and detect abnormal patterns that may indicate
                    malfunctions.
                  </p>
                </div>

                <CodeBlock
                  code={`import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal

# Generate synthetic sensor data with anomalies
np.random.seed(42)
time = np.linspace(0, 10, 1000)
sampling_rate = len(time) / (time[-1] - time[0])

# Normal operation: smooth sine wave with small noise
normal_signal = np.sin(2 * np.pi * 1 * time) + 0.1 * np.random.randn(len(time))

# Add anomalies
sensor_data = normal_signal.copy()

# Anomaly 1: Sudden spike (mechanical fault)
sensor_data[200:205] += 5

# Anomaly 2: High frequency vibration (bearing issue)
sensor_data[400:500] += 0.5 * np.sin(2 * np.pi * 20 * time[400:500])

# Anomaly 3: Drift (sensor degradation)
sensor_data[700:] += np.linspace(0, 2, len(sensor_data[700:]))

# Anomaly 4: Missing data
sensor_data[600:620] = np.nan

print("Sensor Data Analysis")
print("="*60)
print(f"Data points: {len(sensor_data)}")
print(f"Sampling rate: {sampling_rate:.2f} Hz")
print(f"Duration: {time[-1]:.2f} seconds")

# Wavelet-based anomaly detection
def detect_anomalies_wavelet(data, wavelet='db4', level=5):
    """
    Detect anomalies using wavelet decomposition
    
    Returns:
    - anomaly_scores: array of anomaly scores for each point
    - threshold: calculated threshold for anomaly detection
    - anomalies: boolean array marking anomalies
    """
    # Handle NaN values
    data_clean = data.copy()
    nan_mask = np.isnan(data)
    if nan_mask.any():
        # Interpolate NaN values for wavelet analysis
        data_clean[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            data[~nan_mask]
        )
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data_clean, wavelet, level=level)
    
    # Reconstruct detail coefficients at each level
    details = []
    for i in range(1, len(coeffs)):
        # Zero out approximation and other details
        coeffs_temp = [np.zeros_like(c) for c in coeffs]
        coeffs_temp[i] = coeffs[i]
        detail = pywt.waverec(coeffs_temp, wavelet)
        # Ensure same length as input
        detail = detail[:len(data)]
        details.append(detail)
    
    # Calculate anomaly score as weighted sum of detail energies
    anomaly_score = np.zeros(len(data))
    weights = np.array([2**i for i in range(len(details))])
    weights = weights / weights.sum()
    
    for i, detail in enumerate(details):
        # Use local energy in sliding window
        window_size = 20
        detail_energy = np.convolve(detail**2, 
                                    np.ones(window_size)/window_size, 
                                    mode='same')
        anomaly_score += weights[i] * detail_energy
    
    # Calculate threshold using MAD (Median Absolute Deviation)
    median = np.median(anomaly_score)
    mad = np.median(np.abs(anomaly_score - median))
    threshold = median + 3 * mad  # 3-sigma rule
    
    # Mark anomalies
    anomalies = anomaly_score > threshold
    
    # Also mark NaN regions as anomalies
    anomalies[nan_mask] = True
    
    return anomaly_score, threshold, anomalies, details

# Detect anomalies
anomaly_scores, threshold, anomalies, details = detect_anomalies_wavelet(sensor_data)

# Find anomaly regions
def find_anomaly_regions(anomalies, min_duration=5):
    """Group consecutive anomaly points into regions"""
    regions = []
    in_region = False
    start = 0
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly and not in_region:
            start = i
            in_region = True
        elif not is_anomaly and in_region:
            if i - start >= min_duration:
                regions.append((start, i))
            in_region = False
    
    if in_region and len(anomalies) - start >= min_duration:
        regions.append((start, len(anomalies)))
    
    return regions

anomaly_regions = find_anomaly_regions(anomalies)

print(f"\\nAnomalies Detected: {len(anomaly_regions)}")

# Visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

# Plot 1: Original sensor data with anomalies highlighted
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, sensor_data, 'b-', label='Sensor Data', alpha=0.7)
for start, end in anomaly_regions:
    ax1.axvspan(time[start], time[end], alpha=0.3, color='red')
ax1.set_ylabel('Amplitude')
ax1.set_title('Sensor Data with Detected Anomalies (Red Regions)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Anomaly score
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time, anomaly_scores, 'g-', label='Anomaly Score')
ax2.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
ax2.fill_between(time, 0, anomaly_scores, where=anomalies, 
                 alpha=0.3, color='red', label='Anomalies')
ax2.set_ylabel('Score')
ax2.set_title('Anomaly Detection Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3-6: Wavelet details at different scales
for i, detail in enumerate(details[:4]):
    ax = fig.add_subplot(gs[2 + i//2, i%2])
    ax.plot(time, detail, label=f'Detail Level {i+1}')
    for start, end in anomaly_regions:
        ax.axvspan(time[start], time[end], alpha=0.2, color='red')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Wavelet Detail Level {i+1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if i >= 2:
        ax.set_xlabel('Time (s)')

plt.suptitle('Wavelet-based Anomaly Detection in Sensor Data', 
             fontsize=14, y=0.995)
plt.show()

# Generate detailed anomaly report
print("\\n" + "="*60)
print("ANOMALY DETECTION REPORT")
print("="*60)

for idx, (start, end) in enumerate(anomaly_regions, 1):
    duration = (time[end-1] - time[start]) if end < len(time) else (time[-1] - time[start])
    max_score = np.max(anomaly_scores[start:end])
    mean_value = np.nanmean(sensor_data[start:end])
    
    print(f"\\nAnomaly #{idx}:")
    print(f"  Time Range: {time[start]:.2f}s - {time[end-1] if end < len(time) else time[-1]:.2f}s")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Max Anomaly Score: {max_score:.4f}")
    print(f"  Mean Value: {mean_value:.4f}")
    
    # Classify anomaly type based on characteristics
    value_change = abs(mean_value - np.nanmean(normal_signal))
    detail_energy = [np.mean(d[start:end]**2) for d in details]
    high_freq_energy = sum(detail_energy[:2])
    
    if np.isnan(sensor_data[start:end]).any():
        anomaly_type = "Missing Data"
        severity = "HIGH"
    elif value_change > 2:
        anomaly_type = "Sudden Spike/Fault"
        severity = "CRITICAL"
    elif high_freq_energy > 0.5:
        anomaly_type = "High Frequency Vibration"
        severity = "MEDIUM"
    elif duration > 1.0:
        anomaly_type = "Drift/Degradation"
        severity = "MEDIUM"
    else:
        anomaly_type = "Transient Event"
        severity = "LOW"
    
    print(f"  Type: {anomaly_type}")
    print(f"  Severity: {severity}")

print("\\n" + "="*60)
print("SUMMARY STATISTICS:")
print("="*60)
print(f"Total anomalies detected: {len(anomaly_regions)}")
print(f"Total anomalous time: {sum([time[end-1]-time[start] for start, end in anomaly_regions]):.2f}s")
print(f"Percentage of anomalous data: {100 * np.sum(anomalies) / len(anomalies):.2f}%")
print(f"Detection threshold: {threshold:.4f}")

print("\\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("• Schedule immediate inspection for CRITICAL anomalies")
print("• Monitor MEDIUM severity events for recurring patterns")
print("• Log LOW severity events for trend analysis")
print("•Review sensor calibration if drift detected")
print("• Check mechanical components if vibration detected")
print("• Verify data acquisition system for missing data issues")
print("="*60)

# Statistical analysis
normal_data = sensor_data[~anomalies]
anomalous_data = sensor_data[anomalies]

print("\\n" + "="*60)
print("STATISTICAL COMPARISON:")
print("="*60)
print("\\nNormal Operation:")
print(f"  Mean: {np.nanmean(normal_data):.4f}")
print(f"  Std Dev: {np.nanstd(normal_data):.4f}")
print(f"  Min: {np.nanmin(normal_data):.4f}")
print(f"  Max: {np.nanmax(normal_data):.4f}")

print("\\nAnomalous Operation:")
print(f"  Mean: {np.nanmean(anomalous_data):.4f}")
print(f"  Std Dev: {np.nanstd(anomalous_data):.4f}")
print(f"  Min: {np.nanmin(anomalous_data):.4f}")
print(f"  Max: {np.nanmax(anomalous_data):.4f}")

print("="*60)

# Real-time monitoring simulation
print("\\n" + "="*60)
print("REAL-TIME MONITORING MODE:")
print("="*60)
print("Simulating real-time anomaly detection...")

# Process data in chunks
chunk_size = 50
for i in range(0, len(sensor_data), chunk_size):
    chunk = sensor_data[max(0, i-100):i+chunk_size]  # Include history
    time_chunk = time[max(0, i-100):i+chunk_size]
    
    if len(chunk) > 100:  # Need enough data for wavelet
        scores, thresh, anom, _ = detect_anomalies_wavelet(chunk)
        
        # Check current chunk for anomalies
        current_anom = anom[-chunk_size:] if len(anom) >= chunk_size else anom
        
        if np.any(current_anom):
            current_time = time_chunk[-chunk_size:] if len(time_chunk) >= chunk_size else time_chunk
            print(f"  ⚠ ALERT at t={current_time[0]:.2f}s - Anomaly detected!")

print("\\nReal-time monitoring simulation complete.")
print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task4">
              <AccordionTrigger className="text-lg font-semibold">
                Task 4: Lane Detection Using Hough Transform
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 mb-2">Scenario:</p>
                  <p className="text-green-800 text-sm">
                    Working on autonomous vehicle navigation, detect lane
                    markings on the road using Hough Line Transformation to
                    assist lane-keeping algorithms.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load road image or video frame
road_image = cv2.imread('road.jpg')
road_rgb = cv2.cvtColor(road_image, cv2.COLOR_BGR2RGB)
height, width = road_rgb.shape[:2]

# Define region of interest (ROI) - trapezoid shape
def region_of_interest(img):
    """Define ROI as trapezoid covering lane area"""
    mask = np.zeros_like(img)
    
    # Define trapezoid vertices
    # Adjust these based on camera position
    vertices = np.array([[
        (int(width * 0.1), height),           # Bottom left
        (int(width * 0.4), int(height * 0.6)), # Top left
        (int(width * 0.6), int(height * 0.6)), # Top right
        (int(width * 0.9), height)             # Bottom right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, vertices

# Convert to grayscale
gray = cv2.cvtColor(road_rgb, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Enhance lane markings using white/yellow color filtering
hsv = cv2.cvtColor(road_image, cv2.COLOR_BGR2HSV)

# White lane detection
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 30, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# Yellow lane detection
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Combine masks
lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

# Apply to grayscale
masked_gray = cv2.bitwise_and(blurred, blurred, mask=lane_mask)

# Canny edge detection
edges = cv2.Canny(masked_gray, 50, 150)

# Apply region of interest
roi_edges, roi_vertices = region_of_interest(edges)

# Hough Line Transform
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi/180,
    threshold=50,
    minLineLength=100,
    maxLineGap=50
)

# Separate left and right lane lines
def separate_lanes(lines):
    """Separate detected lines into left and right lanes"""
    left_lines = []
    right_lines = []
    
    if lines is None:
        return left_lines, right_lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical or horizontal lines
        if x2 - x1 == 0:
            continue
        
        # Calculate slope
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter by slope and position
        if slope < -0.5 and x1 < width // 2:  # Left lane
            left_lines.append(line[0])
        elif slope > 0.5 and x1 > width // 2:  # Right lane
            right_lines.append(line[0])
    
    return left_lines, right_lines

left_lines, right_lines = separate_lanes(lines)

# Average and extrapolate lane lines
def average_lines(lines, y_min, y_max):
    """Average multiple lines and extrapolate to full lane"""
    if not lines:
        return None
    
    # Fit line to all points
    x_coords = []
    y_coords = []
    
    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # Fit polynomial (degree 1 = line)
    if len(x_coords) > 0:
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Calculate x coordinates for y_min and y_max
        x_min = int(np.polyval(poly, y_min))
        x_max = int(np.polyval(poly, y_max))
        
        return [x_min, y_min, x_max, y_max]
    
    return None

# Extrapolate lanes
y_min = int(height * 0.6)
y_max = height

left_lane = average_lines(left_lines, y_min, y_max)
right_lane = average_lines(right_lines, y_min, y_max)

# Draw lane lines
def draw_lanes(img, left_line, right_line, color=(0, 255, 0), thickness=10):
    """Draw detected lane lines on image"""
    line_img = np.zeros_like(img)
    
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

# Fill lane area
def fill_lane(img, left_line, right_line):
    """Fill detected lane area"""
    lane_img = np.zeros_like(img)
    
    if left_line is not None and right_line is not None:
        x1_l, y1_l, x2_l, y2_l = left_line
        x1_r, y1_r, x2_r, y2_r = right_line
        
        # Create polygon
        pts = np.array([[x1_l, y1_l], [x2_l, y2_l], 
                       [x2_r, y2_r], [x1_r, y1_r]], dtype=np.int32)
        
        cv2.fillPoly(lane_img, [pts], (0, 255, 0))
    
    return cv2.addWeighted(img, 0.8, lane_img, 0.3, 0)

# Create visualizations
img_with_lines = draw_lanes(road_rgb.copy(), left_lane, right_lane)
img_with_fill = fill_lane(road_rgb.copy(), left_lane, right_lane)

# Draw all detected line segments
img_all_lines = road_rgb.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_all_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Draw ROI
img_roi = road_rgb.copy()
cv2.polylines(img_roi, roi_vertices, True, (255, 255, 0), 3)

# Display results
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

axes[0, 0].imshow(road_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(lane_mask, cmap='gray')
axes[0, 1].set_title('Lane Color Mask')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges, cmap='gray')
axes[0, 2].set_title('Canny Edges')
axes[0, 2].axis('off')

axes[1, 0].imshow(roi_edges, cmap='gray')
axes[1, 0].set_title('ROI Edges')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_roi)
axes[1, 1].set_title('Region of Interest')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_all_lines)
axes[1, 2].set_title(f'All Hough Lines ({len(lines) if lines is not None else 0})')
axes[1, 2].axis('off')

axes[2, 0].imshow(img_with_lines)
axes[2, 0].set_title('Detected Lane Lines')
axes[2, 0].axis('off')

axes[2, 1].imshow(img_with_fill)
axes[2, 1].set_title('Lane Area Highlighted')
axes[2, 1].axis('off')

# Calculate lane metrics
if left_lane is not None and right_lane is not None:
    # Lane width at bottom
    x1_l, _, _, y_l = left_lane
    x1_r, _, _, y_r = right_lane
    lane_width = abs(x1_r - x1_l)
    
    # Lane center
    lane_center = (x1_l + x1_r) // 2
    image_center = width // 2
    center_offset = lane_center - image_center
    
    # Draw metrics
    img_metrics = img_with_fill.copy()
    cv2.line(img_metrics, (lane_center, height-50), 
             (lane_center, height-10), (255, 0, 0), 3)
    cv2.line(img_metrics, (image_center, height-50), 
             (image_center, height-10), (0, 0, 255), 3)
    
    cv2.putText(img_metrics, f'Lane Width: {lane_width}px', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_metrics, f'Center Offset: {center_offset}px', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    axes[2, 2].imshow(img_metrics)
    axes[2, 2].set_title('Lane Metrics')
    axes[2, 2].axis('off')
else:
    axes[2, 2].text(0.5, 0.5, 'Lane Not Detected', 
                    ha='center', va='center', fontsize=16, color='red')
    axes[2, 2].axis('off')

plt.tight_layout()
plt.show()

# Print analysis
print("="*60)
print("LANE DETECTION ANALYSIS")
print("="*60)

print(f"\\nDetection Results:")
print(f"  Total lines detected: {len(lines) if lines is not None else 0}")
print(f"  Left lane lines: {len(left_lines)}")
print(f"  Right lane lines: {len(right_lines)}")
print(f"  Left lane detected: {'Yes' if left_lane is not None else 'No'}")
print(f"  Right lane detected: {'Yes' if right_lane is not None else 'No'}")

if left_lane is not None and right_lane is not None:
    print(f"\\nLane Metrics:")
    print(f"  Lane width: {lane_width} pixels")
    print(f"  Lane center: {lane_center} pixels")
    print(f"  Image center: {image_center} pixels")
    print(f"  Center offset: {center_offset} pixels")
    
    # Steering recommendation
    if abs(center_offset) < 20:
        steering = "Centered"
    elif center_offset < 0:
        steering = "Steer Right"
    else:
        steering = "Steer Left"
    
    print(f"\\nSteering Recommendation: {steering}")

print("\\n" + "="*60)
print("LANE KEEPING GUIDANCE:")
print("="*60)
print("• Lane center offset indicates vehicle position")
print("• Negative offset: Vehicle is left of center")
print("• Positive offset: Vehicle is right of center")
print("• Adjust steering to minimize offset")
print("="*60)`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task5">
              <AccordionTrigger className="text-lg font-semibold">
                Task 5: Coin Detection and Counting
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="bg-yellow-50 p-4 rounded-lg">
                  <p className="font-semibold text-yellow-900 mb-2">
                    Scenario:
                  </p>
                  <p className="text-yellow-800 text-sm">
                    Identify and count coins in images using Hough Circle
                    Transformation. Handle coins of various sizes and positions
                    for automatic counting systems.
                  </p>
                </div>

                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load coin image
coins_img = cv2.imread('coins.jpg')
coins_rgb = cv2.cvtColor(coins_img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(coins_img, cv2.COLOR_BGR2GRAY)

# Preprocessing
# Apply CLAHE for better contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Reduce noise
blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=15,
    maxRadius=100
)

# Process detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    circles_list = circles[0, :]
    
    # Sort by size (radius)
    circles_sorted = sorted(circles_list, key=lambda c: c[2], reverse=True)
    
    # Classify coins by size
    radii = [c[2] for c in circles_sorted]
    
    # Use k-means to cluster coin sizes
    from sklearn.cluster import KMeans
    
    # Determine number of coin types (2-4 typical)
    n_types = min(4, len(set(radii)))
    
    if len(radii) > 1 and n_types > 1:
        kmeans = KMeans(n_clusters=n_types, random_state=42, n_init=10)
        size_labels = kmeans.fit_predict(np.array(radii).reshape(-1, 1))
        size_centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
    else:
        size_labels = np.zeros(len(radii))
        size_centers = [np.mean(radii)] if radii else []
    
    # Assign coin types
    coin_types = ['Quarter', 'Nickel', 'Dime', 'Penny']
    coin_values = [0.25, 0.05, 0.10, 0.01]
    
    coin_data = []
    for idx, (x, y, r) in enumerate(circles_sorted):
        coin_type_idx = int(size_labels[idx])
        coin_info = {
            'id': idx + 1,
            'center': (x, y),
            'radius': r,
            'type_idx': coin_type_idx,
            'type': coin_types[coin_type_idx] if coin_type_idx < len(coin_types) else f'Type {coin_type_idx+1}',
            'value': coin_values[coin_type_idx] if coin_type_idx < len(coin_values) else 0
        }
        coin_data.append(coin_info)
    
    # Visualizations
    img_detected = coins_rgb.copy()
    img_labeled = coins_rgb.copy()
    img_types = coins_rgb.copy()
    
    # Color map for different coin types
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for coin in coin_data:
        x, y, r = coin['center'][0], coin['center'][1], coin['radius']
        color = colors[coin['type_idx'] % len(colors)]
        
        # Draw on detected image
        cv2.circle(img_detected, (x, y), r, (0, 255, 0), 3)
        cv2.circle(img_detected, (x, y), 2, (255, 0, 0), 3)
        
        # Draw on labeled image
        cv2.circle(img_labeled, (x, y), r, (0, 255, 0), 2)
        cv2.putText(img_labeled, str(coin['id']), (x-10, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw on type image
        cv2.circle(img_types, (x, y), r, color, 3)
        cv2.circle(img_types, (x, y), 2, color, -1)
        cv2.putText(img_types, coin['type'][:3], (x-15, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create histogram of coin sizes
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 6))
    ax_hist.hist(radii, bins=20, edgecolor='black', alpha=0.7)
    for center in size_centers:
        ax_hist.axvline(x=center, color='r', linestyle='--', linewidth=2)
    ax_hist.set_xlabel('Radius (pixels)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Coin Sizes')
    ax_hist.grid(True, alpha=0.3)
    plt.show()
    
    # Display main results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(coins_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title('Enhanced (CLAHE)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Blurred')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_detected)
    axes[1, 0].set_title(f'Detected Circles ({len(circles_sorted)})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_labeled)
    axes[1, 1].set_title('Labeled Coins')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_types)
    axes[1, 2].set_title('Coins by Type')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Generate report
    print("="*70)
    print("COIN COUNTING REPORT")
    print("="*70)
    
    print(f"\\nTotal Coins Detected: {len(coin_data)}")
    print(f"Coin Types Identified: {n_types}")
    
    # Count by type
    type_counts = {}
    for coin in coin_data:
        coin_type = coin['type']
        type_counts[coin_type] = type_counts.get(coin_type, 0) + 1
    
    print(f"\\nBreakdown by Type:")
    for coin_type, count in sorted(type_counts.items()):
        print(f"  {coin_type}: {count}")
    
    # Calculate total value
    total_value = sum(coin['value'] for coin in coin_data)
    print(f"\\nEstimated Total Value: {total_value:.2f}")
    
    print(f"\\nDetailed Coin Information:")
    print("-"*70)
    print(f"{'ID':<4} {'Type':<10} {'Position':<15} {'Radius':<8} {'Value':<8}")
    print("-"*70)
    
    for coin in coin_data:
        pos_str = f"({coin['center'][0]},{coin['center'][1]})"
        val_str = f"{coin['value']:.2f}"
        print(f"{coin['id']:<4} {coin['type']:<10} {pos_str:<15} "
              f"{coin['radius']:<8} {val_str:<8}")
    
    print("-"*70)
    
    # Statistics
    print(f"\\nStatistics:")
    print(f"  Average radius: {np.mean(radii):.2f} pixels")
    print(f"  Std deviation: {np.std(radii):.2f} pixels")
    print(f"  Largest coin: {max(radii)} pixels (ID {coin_data[0]['id']})")
    print(f"  Smallest coin: {min(radii)} pixels (ID {coin_data[-1]['id']})")
    
    print("\\n" + "="*70)

else:
    print("No coins detected!")
    
    # Show preprocessing steps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(coins_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(blurred, cmap='gray')
    axes[2].set_title('Blurred')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\\nTroubleshooting Tips:")
    print("- Adjust Hough Circle parameters")
    print("- Check image quality and lighting")
    print("- Verify coin sizes are within minRadius/maxRadius")
    print("- Try different preprocessing techniques")`}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">
              🎓 Advanced Techniques Summary
            </h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
              <div>
                <p className="font-semibold mb-2">Wavelet Transform:</p>
                <ul className="space-y-1">
                  <li>• Multi-scale time-frequency analysis</li>
                  <li>• Excellent for denoising and compression</li>
                  <li>• Detects transient events in signals</li>
                  <li>• Applications: JPEG2000, anomaly detection</li>
                </ul>
              </div>
              <div>
                <p className="font-semibold mb-2">Hough Transform:</p>
                <ul className="space-y-1">
                  <li>• Robust shape detection (lines, circles)</li>
                  <li>• Handles partial/occluded shapes</li>
                  <li>• Parameter space voting mechanism</li>
                  <li>• Applications: Lane detection, coin counting</li>
                </ul>
              </div>
              <div>
                <p className="font-semibold mb-2">SIFT Features:</p>
                <ul className="space-y-1">
                  <li>• Scale and rotation invariant</li>
                  <li>• 128-dimensional descriptors</li>
                  <li>• Robust to illumination changes</li>
                  <li>• Applications: Object recognition, stitching</li>
                </ul>
              </div>
              <div>
                <p className="font-semibold mb-2">Best Practices:</p>
                <ul className="space-y-1">
                  <li>• Always preprocess images (blur, enhance)</li>
                  <li>• Tune parameters for specific applications</li>
                  <li>• Combine techniques for robust solutions</li>
                  <li>• Validate results with ground truth</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
