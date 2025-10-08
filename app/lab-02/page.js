import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { Palette, Code, CheckCircle } from "lucide-react";

export default function Lab02() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Lab 02: Image Enhancement & Transformations
        </h1>
        <p className="text-slate-600 text-lg">
          Point processing, pixel transformations, color spaces, and histogram
          equalization
        </p>
      </div>

      <Tabs defaultValue="concepts" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="concepts">
            <Palette className="w-4 h-4 mr-2" />
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
          <ConceptCard title="Digital Images and Representation">
            <div className="space-y-4 text-slate-700">
              <p>
                Digital images are visual representations stored as matrices or
                grids of pixels. Each pixel contains color and intensity
                information.
              </p>

              <div className="bg-slate-100 p-4 rounded-lg">
                <p className="font-semibold mb-2">RGB Representation:</p>
                <p className="text-sm mb-2">
                  Each pixel is represented as (R, G, B) where each component
                  ranges from 0 to 255.
                </p>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div className="bg-red-500 text-white p-2 rounded text-center">
                    (255, 0, 0) Red
                  </div>
                  <div className="bg-green-500 text-white p-2 rounded text-center">
                    (0, 255, 0) Green
                  </div>
                  <div className="bg-blue-500 text-white p-2 rounded text-center">
                    (0, 0, 255) Blue
                  </div>
                  <div className="bg-white border p-2 rounded text-center">
                    (255, 255, 255) White
                  </div>
                  <div className="bg-gray-500 text-white p-2 rounded text-center">
                    (128, 128, 128) Gray
                  </div>
                  <div className="bg-black text-white p-2 rounded text-center">
                    (0, 0, 0) Black
                  </div>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Point Processing">
            <div className="space-y-4 text-slate-700">
              <p>
                Point processing modifies individual pixel values without
                considering neighboring pixels. It's simple yet powerful for
                image enhancement.
              </p>

              <Accordion type="single" collapsible>
                <AccordionItem value="brightness">
                  <AccordionTrigger>Brightness Adjustment</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">
                      Add or subtract a constant value to all pixels:
                    </p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      New Pixel = Old Pixel + Brightness Factor
                    </code>
                    <p className="mt-2 text-sm">
                      Positive values brighten, negative values darken the
                      image.
                    </p>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="contrast">
                  <AccordionTrigger>Contrast Enhancement</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">Stretch the range of pixel values:</p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      New Pixel = α × Old Pixel + β
                    </code>
                    <p className="mt-2 text-sm">
                      α controls contrast, β controls brightness. α &gt; 1
                      increases contrast.
                    </p>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="threshold">
                  <AccordionTrigger>Thresholding</AccordionTrigger>
                  <AccordionContent>
                    <p className="mb-2">Convert to binary image:</p>
                    <code className="block bg-slate-900 text-white p-3 rounded">
                      New Pixel = 255 if Old Pixel &gt; Threshold else 0
                    </code>
                    <p className="mt-2 text-sm">
                      Used for image segmentation and object detection.
                    </p>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </ConceptCard>

          <ConceptCard title="Pixel Transformations">
            <div className="space-y-4 text-slate-700">
              <p>
                Pixel transformations apply mathematical functions to enhance or
                alter image appearance.
              </p>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Logarithmic Transformation
                  </h4>
                  <code className="text-sm block bg-white p-2 rounded mb-2">
                    New = c × log(1 + Old)
                  </code>
                  <p className="text-sm text-purple-800">
                    Enhances darker regions while compressing brighter areas.
                    Reveals details in shadows.
                  </p>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    Gamma Correction
                  </h4>
                  <code className="text-sm block bg-white p-2 rounded mb-2">
                    New = c × (Old ^ γ)
                  </code>
                  <p className="text-sm text-blue-800">
                    γ &lt; 1 brightens midtones, γ &gt; 1 darkens them. Adjusts
                    overall brightness and contrast.
                  </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Negative Image
                  </h4>
                  <code className="text-sm block bg-white p-2 rounded mb-2">
                    New = 255 - Old
                  </code>
                  <p className="text-sm text-green-800">
                    Inverts pixel values, creating a photographic negative
                    effect.
                  </p>
                </div>

                <div className="bg-orange-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-orange-900 mb-2">
                    Contrast Stretching
                  </h4>
                  <code className="text-sm block bg-white p-2 rounded mb-2">
                    New = (Old - Min) × 255 / (Max - Min)
                  </code>
                  <p className="text-sm text-orange-800">
                    Stretches pixel values to full range [0, 255] for maximum
                    contrast.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Color Spaces">
            <div className="space-y-4 text-slate-700">
              <p>
                Color spaces are standardized systems for representing colors.
                Different spaces capture various aspects like luminance,
                chrominance, and perceptual uniformity.
              </p>

              <div className="space-y-3">
                <div className="border-l-4 border-red-500 pl-4 py-2">
                  <h4 className="font-semibold">RGB (Red-Green-Blue)</h4>
                  <p className="text-sm">
                    Additive color model used in displays. Direct correspondence
                    with screens and cameras.
                  </p>
                </div>

                <div className="border-l-4 border-cyan-500 pl-4 py-2">
                  <h4 className="font-semibold">HSV (Hue-Saturation-Value)</h4>
                  <p className="text-sm">
                    Separates color (hue) from intensity (value). Useful for
                    color-based segmentation and detection.
                  </p>
                </div>

                <div className="border-l-4 border-purple-500 pl-4 py-2">
                  <h4 className="font-semibold">LAB (CIELAB)</h4>
                  <p className="text-sm">
                    Perceptually uniform color space. L = luminance, A =
                    green-red, B = blue-yellow. Great for color correction.
                  </p>
                </div>

                <div className="border-l-4 border-yellow-500 pl-4 py-2">
                  <h4 className="font-semibold">YUV/YCbCr</h4>
                  <p className="text-sm">
                    Separates luminance (Y) from chrominance (UV). Efficient for
                    video compression and transmission.
                  </p>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Histogram Equalization">
            <div className="space-y-4 text-slate-700">
              <p>
                Histogram equalization redistributes pixel intensities to
                enhance contrast. It applies a mapping function based on the
                image's cumulative histogram.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  How it works:
                </p>
                <ol className="list-decimal list-inside space-y-2 text-blue-800">
                  <li>Calculate histogram of pixel intensities</li>
                  <li>Compute cumulative distribution function (CDF)</li>
                  <li>Normalize CDF to range [0, 255]</li>
                  <li>Map each pixel to its equalized value</li>
                </ol>
              </div>

              <p className="text-sm">
                Result: Enhanced contrast with more uniformly distributed pixel
                values, making details more visible.
              </p>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="Brightness Adjustment">
            <div className="space-y-4">
              <p className="text-slate-700">
                Increase or decrease overall image brightness by adding a
                constant value to all pixels.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')

# Define brightness adjustment factor
brightness_factor = 50  # Positive = brighter, Negative = darker

# Apply brightness adjustment
# np.clip ensures values stay in valid range [0, 255]
brightened_image = np.clip(image + brightness_factor, 0, 255).astype(np.uint8)

# Display results
stacked = np.hstack([image, brightened_image])
cv2.imshow('Original vs Brightened', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Contrast Enhancement">
            <div className="space-y-4">
              <p className="text-slate-700">
                Enhance contrast using the formula: New = α × Old + β, where α
                controls contrast and β controls brightness.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread('image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define contrast enhancement parameters
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 50    # Brightness control

# Apply contrast enhancement
enhanced_image = np.clip(alpha * gray_image + beta, 0, 255).astype(np.uint8)

# Display results
stacked = np.hstack([gray_image, enhanced_image])
cv2.imshow('Original vs Enhanced', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Logarithmic Transformation">
            <div className="space-y-4">
              <p className="text-slate-700">
                Enhance darker regions using logarithmic transformation. Useful
                for revealing details in shadowy areas.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate constant c for scaling
c = 255 / np.log(1 + np.max(gray_image))

# Apply logarithmic transformation
log_transformed = c * np.log(1 + gray_image)

# Convert to uint8 for display
log_transformed = np.array(log_transformed, dtype=np.uint8)

# Display results
stacked = np.hstack([gray_image, log_transformed])
cv2.imshow('Original vs Log Transformed', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                <p className="text-sm text-yellow-800">
                  <strong>Tip:</strong> Adding 1 to pixel values prevents log(0)
                  errors. The constant c scales output to [0, 255].
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Gamma Correction">
            <div className="space-y-4">
              <p className="text-slate-700">
                Adjust brightness and contrast using power-law transformation.
                Gamma &lt; 1 brightens, gamma &gt; 1 darkens.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define gamma value
gamma = 0.5  # < 1 brightens, > 1 darkens

# Apply power-law transformation
gamma_corrected = np.power(gray_image / 255.0, gamma)

# Normalize to [0, 255]
gamma_corrected = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)

# Display results
stacked = np.hstack([gray_image, gamma_corrected])
cv2.imshow('Original vs Gamma Corrected', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Histogram Equalization">
            <div className="space-y-4">
              <p className="text-slate-700">
                Enhance image contrast by redistributing pixel intensities using
                histogram equalization.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display results
stacked = np.hstack([gray_image, equalized_image])
cv2.imshow('Original vs Equalized', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Contrast Stretching (Piecewise Linear)">
            <div className="space-y-4">
              <p className="text-slate-700">
                Stretch pixel values from their current range to the full [0,
                255] range for maximum contrast.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Get min and max pixel values
input_min = np.min(gray_image)
input_max = np.max(gray_image)
output_min = 0
output_max = 255

# Apply contrast stretching
stretched = ((gray_image - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min

# Convert to uint8
stretched = np.array(stretched, dtype=np.uint8)

# Display results
stacked = np.hstack([gray_image, stretched])
cv2.imshow('Original vs Stretched', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Color Space Conversions">
            <div className="space-y-4">
              <p className="text-slate-700">
                Convert between different color spaces to separate color
                information from intensity or for specific applications.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load color image
color_image = cv2.imread('image.jpg')

# Convert to different color spaces
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
ycrcb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)

# Split channels for visualization
h, s, v = cv2.split(hsv_image)
l, a, b = cv2.split(lab_image)
y, cr, cb = cv2.split(ycrcb_image)

# Display HSV channels
cv2.imshow('Hue', h)
cv2.imshow('Saturation', s)
cv2.imshow('Value', v)

# Display LAB channels
cv2.imshow('Lightness', l)
cv2.imshow('A channel', a)
cv2.imshow('B channel', b)

cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Color Balance Adjustment">
            <div className="space-y-4">
              <p className="text-slate-700">
                Adjust color balance by scaling individual color channels
                independently.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load color image
color_image = cv2.imread('image.jpg')

# Define color balance adjustment factors for BGR channels
blue_scale = 1.2   # Increase blue
green_scale = 1.0  # Keep green unchanged
red_scale = 0.9    # Decrease red

# Apply color balance adjustment
adjusted_image = np.clip(
    color_image * [blue_scale, green_scale, red_scale], 
    0, 255
).astype(np.uint8)

# Display results
stacked = np.hstack([color_image, adjusted_image])
cv2.imshow('Original vs Adjusted', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Apply Colormap">
            <div className="space-y-4">
              <p className="text-slate-700">
                Apply pseudocolor to grayscale images using colormaps for better
                visualization.
              </p>
              <CodeBlock
                code={`import cv2

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply different colormaps
jet_colormap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
hot_colormap = cv2.applyColorMap(gray_image, cv2.COLORMAP_HOT)
rainbow_colormap = cv2.applyColorMap(gray_image, cv2.COLORMAP_RAINBOW)
cool_colormap = cv2.applyColorMap(gray_image, cv2.COLORMAP_COOL)

# Display results
cv2.imshow('JET', jet_colormap)
cv2.imshow('HOT', hot_colormap)
cv2.imshow('RAINBOW', rainbow_colormap)
cv2.imshow('COOL', cool_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-blue-50 p-4 rounded mt-4">
                <p className="font-semibold text-blue-900 mb-2">
                  Available Colormaps:
                </p>
                <div className="grid grid-cols-2 gap-2 text-sm text-blue-800">
                  <div>• COLORMAP_JET</div>
                  <div>• COLORMAP_HOT</div>
                  <div>• COLORMAP_RAINBOW</div>
                  <div>• COLORMAP_COOL</div>
                  <div>• COLORMAP_AUTUMN</div>
                  <div>• COLORMAP_WINTER</div>
                  <div>• COLORMAP_SPRING</div>
                  <div>• COLORMAP_SUMMER</div>
                </div>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Color Inversion">
            <div className="space-y-4">
              <p className="text-slate-700">
                Create negative image by inverting all color channels.
              </p>
              <CodeBlock
                code={`import cv2

# Load color image
color_image = cv2.imread('image.jpg')

# Invert colors using bitwise_not
inverted_image = cv2.bitwise_not(color_image)

# Alternative method using arithmetic
# inverted_image = 255 - color_image

# Display results
stacked = np.hstack([color_image, inverted_image])
cv2.imshow('Original vs Inverted', stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Binary Thresholding">
            <div className="space-y-4">
              <p className="text-slate-700">
                Convert grayscale image to binary (black and white) based on
                threshold value.
              </p>
              <CodeBlock
                code={`import cv2

# Load grayscale image
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
threshold_value = 128
_, binary_image = cv2.threshold(
    gray_image, 
    threshold_value, 
    255, 
    cv2.THRESH_BINARY
)

# Different thresholding methods
_, binary_inv = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TOZERO)

# Display results
cv2.imshow('Original', gray_image)
cv2.imshow('Binary', binary_image)
cv2.imshow('Binary Inverse', binary_inv)
cv2.imshow('Truncate', trunc)
cv2.imshow('To Zero', tozero)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-6 mt-6">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-purple-900 mb-4">
              Lab Tasks
            </h2>
            <p className="text-purple-800 mb-4">
              Complete these real-world medical imaging tasks to apply the
              concepts learned in this lab.
            </p>
          </div>

          <ConceptCard title="Task 1: Medical Image Enhancement">
            <div className="space-y-4">
              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <h3 className="font-semibold text-blue-900 mb-2">Scenario:</h3>
                <p className="text-blue-800">
                  You're a junior researcher at a medical imaging lab. The lab
                  has received grayscale X-ray images with poor contrast and
                  visibility issues. Your task is to enhance these images using
                  various techniques.
                </p>
              </div>

              <div className="space-y-3 text-slate-700">
                <h4 className="font-semibold">Steps to Complete:</h4>
                <ol className="list-decimal list-inside space-y-2 ml-4">
                  <li>
                    <strong>Load and Display:</strong> Load an X-ray image and
                    display it
                  </li>
                  <li>
                    <strong>Contrast Enhancement:</strong> Apply histogram
                    equalization to enhance contrast
                  </li>
                  <li>
                    <strong>Color Mapping:</strong> Convert to color-coded
                    heatmap using "jet" colormap
                  </li>
                  <li>
                    <strong>Color Balance:</strong> Apply color balance
                    adjustment to remove color casts
                  </li>
                  <li>
                    <strong>Logarithmic Transform:</strong> Enhance darker areas
                    for critical details
                  </li>
                  <li>
                    <strong>Power-Law Transform:</strong> Fine-tune contrast
                    using gamma correction
                  </li>
                </ol>
              </div>

              <CodeBlock
                code={`import cv2
import numpy as np

# Load X-ray image
xray = cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Histogram equalization for contrast enhancement
equalized = cv2.equalizeHist(xray)

# 2. Apply JET colormap for better visualization
colormap_jet = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)

# 3. Logarithmic transformation for darker areas
c = 255 / np.log(1 + np.max(equalized))
log_transformed = c * np.log(1 + equalized)
log_transformed = np.array(log_transformed, dtype=np.uint8)

# 4. Power-law transformation (gamma correction)
gamma = 0.7
gamma_corrected = np.power(log_transformed / 255.0, gamma)
gamma_corrected = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)

# Display all stages
cv2.imshow('1. Original X-ray', xray)
cv2.imshow('2. Histogram Equalized', equalized)
cv2.imshow('3. Colormap Applied', colormap_jet)
cv2.imshow('4. Log Transformed', log_transformed)
cv2.imshow('5. Gamma Corrected', gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Task 2: Multi-Modal Medical Image Fusion">
            <div className="space-y-4">
              <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                <h3 className="font-semibold text-green-900 mb-2">Scenario:</h3>
                <p className="text-green-800">
                  You're a senior researcher fusing X-ray and MRI scans for
                  comprehensive diagnostic insights. The challenge is
                  integrating images with varying contrasts and spatial
                  resolutions.
                </p>
              </div>

              <div className="space-y-3 text-slate-700">
                <h4 className="font-semibold">Steps to Complete:</h4>
                <ol className="list-decimal list-inside space-y-2 ml-4">
                  <li>
                    <strong>Load Modalities:</strong> Load X-ray and MRI images
                    of the same region
                  </li>
                  <li>
                    <strong>Histogram Equalization:</strong> Apply to both
                    images separately
                  </li>
                  <li>
                    <strong>Color Mapping & Fusion:</strong> Convert both to
                    colored heatmaps and overlay
                  </li>
                  <li>
                    <strong>Weighted Fusion:</strong> Adjust weighting for
                    optimal feature enhancement
                  </li>
                  <li>
                    <strong>Transformations:</strong> Apply log and power-law
                    transforms to fused image
                  </li>
                  <li>
                    <strong>Analysis:</strong> Compare original and fused images
                  </li>
                </ol>
              </div>

              <CodeBlock
                code={`import cv2
import numpy as np

# Load X-ray and MRI images
xray = cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)
mri = cv2.imread('mri.jpg', cv2.IMREAD_GRAYSCALE)

# Resize MRI to match X-ray dimensions
mri = cv2.resize(mri, (xray.shape[1], xray.shape[0]))

# Apply histogram equalization to both
xray_eq = cv2.equalizeHist(xray)
mri_eq = cv2.equalizeHist(mri)

# Convert to colored heatmaps
xray_colormap = cv2.applyColorMap(xray_eq, cv2.COLORMAP_JET)
mri_colormap = cv2.applyColorMap(mri_eq, cv2.COLORMAP_HOT)

# Multi-modal weighted fusion
weight_xray = 0.6
weight_mri = 0.4
fused_image = cv2.addWeighted(xray_colormap, weight_xray, mri_colormap, weight_mri, 0)

# Convert fused to grayscale for further processing
fused_gray = cv2.cvtColor(fused_image, cv2.COLOR_BGR2GRAY)

# Apply logarithmic transformation
c = 255 / np.log(1 + np.max(fused_gray))
fused_log = c * np.log(1 + fused_gray)
fused_log = np.array(fused_log, dtype=np.uint8)

# Apply gamma correction
gamma = 0.8
fused_gamma = np.power(fused_log / 255.0, gamma)
fused_gamma = np.clip(fused_gamma * 255, 0, 255).astype(np.uint8)

# Display results
cv2.imshow('X-ray Original', xray)
cv2.imshow('MRI Original', mri)
cv2.imshow('X-ray Colormap', xray_colormap)
cv2.imshow('MRI Colormap', mri_colormap)
cv2.imshow('Fused Image', fused_image)
cv2.imshow('Fused + Log Transform', fused_log)
cv2.imshow('Fused + Gamma Correction', fused_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Comparative analysis
print("Image Statistics:")
print(f"X-ray - Mean: {np.mean(xray):.2f}, Std: {np.std(xray):.2f}")
print(f"MRI - Mean: {np.mean(mri):.2f}, Std: {np.std(mri):.2f}")
print(f"Fused - Mean: {np.mean(fused_gray):.2f}, Std: {np.std(fused_gray):.2f}")`}
              />
            </div>
          </ConceptCard>

          <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg">
            <h3 className="font-semibold text-amber-900 mb-3">
              💡 Key Takeaways
            </h3>
            <ul className="space-y-2 text-amber-800">
              <li>
                • <strong>Histogram Equalization:</strong> Best for overall
                contrast enhancement in uniform lighting
              </li>
              <li>
                • <strong>Logarithmic Transform:</strong> Excellent for
                revealing details in dark regions
              </li>
              <li>
                • <strong>Gamma Correction:</strong> Fine-tune brightness while
                maintaining relative intensities
              </li>
              <li>
                • <strong>Color Spaces:</strong> HSV is ideal for color-based
                segmentation, LAB for perceptual uniformity
              </li>
              <li>
                • <strong>Colormaps:</strong> Help visualize intensity
                variations in grayscale medical images
              </li>
              <li>
                • <strong>Multi-modal Fusion:</strong> Combines complementary
                information from different imaging modalities
              </li>
            </ul>
          </div>

          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg">
            <h3 className="font-semibold text-red-900 mb-3">
              ⚠️ Common Pitfalls
            </h3>
            <ul className="space-y-2 text-red-800 text-sm">
              <li>
                • Always use{" "}
                <code className="bg-white px-2 py-1 rounded">np.clip()</code>{" "}
                after transformations to prevent overflow
              </li>
              <li>
                • Convert to{" "}
                <code className="bg-white px-2 py-1 rounded">uint8</code> before
                displaying with OpenCV
              </li>
              <li>
                • Adding 1 in logarithmic transform prevents log(0) errors
              </li>
              <li>• Normalize gamma correction output to [0, 255] range</li>
              <li>• Match image dimensions before fusion operations</li>
              <li>
                • Test different gamma values (0.5-2.0) to find optimal results
              </li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
