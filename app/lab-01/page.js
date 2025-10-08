import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import CodeBlock from "@/components/CodeBlock";
import ConceptCard from "@/components/ConceptCard";
import { BookOpen, Code, CheckCircle } from "lucide-react";

export default function Lab01() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Lab 01: Introduction to Computer Vision
        </h1>
        <p className="text-slate-600 text-lg">
          Fundamentals of computer vision, essential libraries, and basic image
          operations
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
          <ConceptCard title="What is Computer Vision?">
            <div className="space-y-4 text-slate-700">
              <p>
                Computer vision is a multidisciplinary field that enables
                computers to interpret, analyze, and understand visual
                information from the world, similar to human vision. It involves
                developing algorithms, models, and techniques to extract
                meaningful information from images or videos.
              </p>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                <p className="font-semibold text-blue-900 mb-2">
                  Key Components:
                </p>
                <ul className="space-y-2 text-blue-800">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Image Processing:</strong> Manipulating images to
                      enhance quality or extract features
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Pattern Recognition:</strong> Identifying patterns
                      and regularities in data
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>
                      <strong>Machine Learning:</strong> Training algorithms to
                      improve from experience
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Digital Images and Coordinates">
            <div className="space-y-4 text-slate-700">
              <p>
                A digital image is a two-dimensional function{" "}
                <strong>f(x, y)</strong>, where x and y are spatial coordinates,
                and the amplitude of f at any pair of coordinates (x, y)
                represents the intensity or gray level.
              </p>

              <div className="bg-slate-100 p-4 rounded-lg">
                <p className="font-semibold mb-2">Image Representation:</p>
                <ul className="space-y-2 ml-4">
                  <li>
                    <strong>Pixel:</strong> Smallest unit of an image (picture
                    element)
                  </li>
                  <li>
                    <strong>Resolution:</strong> Dimensions expressed in pixels
                    (e.g., 1920×1080)
                  </li>
                  <li>
                    <strong>Grayscale:</strong> Intensity from 0 (black) to 255
                    (white)
                  </li>
                  <li>
                    <strong>RGB:</strong> Color representation using Red, Green,
                    Blue channels
                  </li>
                </ul>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Applications of Computer Vision">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">
                  Autonomous Vehicles
                </h4>
                <p className="text-sm text-green-800">
                  Self-driving cars use CV to perceive environment, recognize
                  obstacles, traffic signs, and make real-time driving
                  decisions.
                </p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-900 mb-2">
                  Medical Imaging
                </h4>
                <p className="text-sm text-purple-800">
                  Assists in diagnosis through image segmentation, tumor
                  detection in X-rays, MRI, CT scans.
                </p>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <h4 className="font-semibold text-orange-900 mb-2">
                  Facial Recognition
                </h4>
                <p className="text-sm text-orange-800">
                  Used for authentication, security, and human-computer
                  interaction like unlocking devices.
                </p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Robotics</h4>
                <p className="text-sm text-blue-800">
                  Robots use CV to navigate, interact with surroundings, grasp
                  objects, and perform visual tasks.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Essential CV Libraries">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="opencv">
                <AccordionTrigger>
                  OpenCV - Most Popular CV Library
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-2 text-slate-700">
                    <p>
                      <strong>Focus:</strong> General computer vision tasks
                    </p>
                    <p>
                      <strong>Language:</strong> C++, Python
                    </p>
                    <p>
                      <strong>Strengths:</strong> High-level abstraction, strong
                      community support, excellent for real-time applications
                    </p>
                    <p className="text-sm bg-slate-100 p-2 rounded">
                      Integration with deep learning through Python bindings
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="pytorch">
                <AccordionTrigger>PyTorch & TorchVision</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-2 text-slate-700">
                    <p>
                      <strong>Focus:</strong> Deep learning for computer vision
                    </p>
                    <p>
                      <strong>Language:</strong> Python
                    </p>
                    <p>
                      <strong>Strengths:</strong> Native integration with
                      PyTorch, excellent for neural networks, strong community
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="scikit">
                <AccordionTrigger>Scikit-Image</AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-2 text-slate-700">
                    <p>
                      <strong>Focus:</strong> Image processing algorithms
                    </p>
                    <p>
                      <strong>Language:</strong> Python
                    </p>
                    <p>
                      <strong>Strengths:</strong> Built on NumPy, Matplotlib
                      integration, medium-level abstraction
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="code" className="space-y-6 mt-6">
          <ConceptCard title="Reading and Displaying Images">
            <div className="space-y-4">
              <p className="text-slate-700">
                The most basic operation in computer vision is loading and
                displaying an image. OpenCV's{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.imread()
                </code>{" "}
                function reads images, and{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.imshow()
                </code>{" "}
                displays them.
              </p>
              <CodeBlock
                code={`import cv2

# Load an image from file
image = cv2.imread('image.png')

# Display the image in a window
cv2.imshow('Image', image)

# Wait for a key press (0 means indefinitely)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Grayscale Conversion">
            <div className="space-y-4">
              <p className="text-slate-700">
                Converting to grayscale reduces the image to a single channel,
                simplifying processing. This is essential for many CV
                algorithms.
              </p>
              <CodeBlock
                code={`import cv2

# Load a color image
image = cv2.imread('image.jpg')

# Convert to grayscale using cvtColor
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                <p className="text-sm text-yellow-800">
                  <strong>Note:</strong> OpenCV loads images in BGR format (not
                  RGB), so use COLOR_BGR2GRAY for proper conversion.
                </p>
              </div>
            </div>
          </ConceptCard>

          <ConceptCard title="Resizing Images">
            <div className="space-y-4">
              <p className="text-slate-700">
                Resizing is crucial for standardizing image dimensions before
                processing or for reducing computational load.
              </p>
              <CodeBlock
                code={`import cv2

# Load an image
image = cv2.imread('image.jpg')

# Resize to specific dimensions (width, height)
new_size = (300, 200)
resized_image = cv2.resize(image, new_size)

# Display both images
cv2.imshow('Original', image)
cv2.imshow('Resized', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Drawing Shapes on Images">
            <div className="space-y-4">
              <p className="text-slate-700">
                OpenCV provides functions to draw geometric shapes, useful for
                annotations and visualizations.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Create a blank black image (300x400 pixels, 3 channels)
image = np.zeros((300, 400, 3), dtype=np.uint8)

# Draw a red rectangle (BGR: 0,0,255)
# Parameters: image, top-left corner, bottom-right corner, color, thickness (-1 = filled)
cv2.rectangle(image, (50, 50), (200, 150), (0, 0, 255), -1)

# Draw a green circle
# Parameters: image, center, radius, color, thickness
cv2.circle(image, (300, 200), 50, (0, 255, 0), -1)

# Display the result
cv2.imshow('Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Filtering (Blur)">
            <div className="space-y-4">
              <p className="text-slate-700">
                Gaussian blur reduces image noise and detail by averaging pixel
                values with a Gaussian kernel.
              </p>
              <CodeBlock
                code={`import cv2

# Load an image
image = cv2.imread('image.jpg')

# Apply Gaussian blur
# Parameters: image, kernel size (must be odd), standard deviation (0 = auto-calculate)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Compare original and blurred
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <p className="text-sm text-slate-600">
                Larger kernel sizes produce stronger blur effects. Common sizes:
                (3,3), (5,5), (7,7).
              </p>
            </div>
          </ConceptCard>

          <ConceptCard title="Image Cropping">
            <div className="space-y-4">
              <p className="text-slate-700">
                Cropping extracts a region of interest (ROI) using NumPy array
                slicing.
              </p>
              <CodeBlock
                code={`import cv2

# Load an image
image = cv2.imread('image.jpg')

# Crop using NumPy slicing: image[y1:y2, x1:x2]
# This crops from row 100-300 and column 150-350
roi = image[100:300, 150:350]

# Display the cropped region
cv2.imshow('Cropped Image', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Adding Text to Images">
            <div className="space-y-4">
              <p className="text-slate-700">
                Use{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.putText()
                </code>{" "}
                to overlay text on images for annotations or labels.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load or create an image
image = cv2.imread('image.png')

# Resize for consistent display
resized_image = cv2.resize(image, (800, 600))

# Add text to the image
text = "Computer Vision Lab"
font = cv2.FONT_HERSHEY_SIMPLEX
# Parameters: image, text, position(x,y), font, scale, color(BGR), thickness
cv2.putText(resized_image, text, (50, 150), font, 1.5, (0, 0, 255), 2)

cv2.imshow('Text on Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Thresholding">
            <div className="space-y-4">
              <p className="text-slate-700">
                Thresholding converts grayscale images to binary (black and
                white) based on a threshold value.
              </p>
              <CodeBlock
                code={`import cv2

# Load a grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
# Parameters: image, threshold value, max value, threshold type
# Pixels > 150 become 255 (white), others become 0 (black)
ret, thresholded_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

cv2.imshow('Original', image)
cv2.imshow('Thresholded', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Rotation">
            <div className="space-y-4">
              <p className="text-slate-700">
                Rotate images using affine transformations with{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.getRotationMatrix2D()
                </code>
                .
              </p>
              <CodeBlock
                code={`import cv2

# Load an image
image = cv2.imread('image.png')

# Get image dimensions
height, width = image.shape[:2]

# Calculate rotation matrix (center, angle, scale)
center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)

# Apply rotation using warpAffine
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Image Blending">
            <div className="space-y-4">
              <p className="text-slate-700">
                Blend two images together using{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.addWeighted()
                </code>{" "}
                or{" "}
                <code className="bg-slate-200 px-2 py-1 rounded">
                  cv2.add()
                </code>
                .
              </p>
              <CodeBlock
                code={`import cv2

# Load two images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Ensure both images are the same size
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Method 1: Simple addition
blended_add = cv2.add(image1, image2)

# Method 2: Weighted blending (alpha blending)
# alpha * img1 + beta * img2 + gamma
alpha = 0.7  # Weight for first image
beta = 0.3   # Weight for second image
blended_weighted = cv2.addWeighted(image1, alpha, image2, beta, 0)

cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Blended (Add)', blended_add)
cv2.imshow('Blended (Weighted)', blended_weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
            </div>
          </ConceptCard>

          <ConceptCard title="Histogram Equalization">
            <div className="space-y-4">
              <p className="text-slate-700">
                Histogram equalization enhances image contrast by redistributing
                pixel intensity values.
              </p>
              <CodeBlock
                code={`import cv2

# Load a grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

cv2.imshow('Original', image)
cv2.imshow('Equalized', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <p className="text-sm text-slate-600">
                This technique is particularly useful for images with poor
                contrast or uneven lighting.
              </p>
            </div>
          </ConceptCard>

          <ConceptCard title="Bitwise Operations">
            <div className="space-y-4">
              <p className="text-slate-700">
                Bitwise operations (AND, OR, XOR, NOT) are useful for masking
                and combining images.
              </p>
              <CodeBlock
                code={`import cv2
import numpy as np

# Load an image
image = cv2.imread('image.png')
height, width, _ = image.shape

# Create a binary mask (white rectangle on black background)
mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (400, 300), 255, -1)

# Perform bitwise operations
bitwise_and = cv2.bitwise_and(image, image, mask=mask)
bitwise_or = cv2.bitwise_or(image, image, mask=mask)
bitwise_xor = cv2.bitwise_xor(image, image, mask=mask)
bitwise_not = cv2.bitwise_not(image, mask=mask)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Mask', mask)
cv2.imshow('AND', bitwise_and)
cv2.imshow('OR', bitwise_or)
cv2.imshow('XOR', bitwise_xor)
cv2.imshow('NOT', bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
              />
              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded mt-4">
                <p className="text-sm text-blue-800">
                  <strong>Use Cases:</strong> Image masking is essential for
                  object extraction, background removal, and region-specific
                  processing.
                </p>
              </div>
            </div>
          </ConceptCard>
        </TabsContent>

        <TabsContent value="tasks" className="space-y-6 mt-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-green-900 mb-4">
              Lab Tasks
            </h2>
            <p className="text-green-800 mb-4">
              Complete these tasks to practice the concepts learned in this lab.
              Each task builds upon the examples above.
            </p>
          </div>

          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="task1">
              <AccordionTrigger className="text-lg font-semibold">
                Task 1: HSV Conversion and Resizing
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Load an image from a file and display it, then convert it to
                  HSV color space instead of grayscale, and finally resize it to
                  half of its original size.
                </p>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="font-semibold text-blue-900 mb-2">Steps:</p>
                  <ol className="list-decimal list-inside space-y-2 text-blue-800">
                    <li>
                      Use{" "}
                      <code className="bg-white px-2 py-1 rounded">
                        cv2.imread()
                      </code>{" "}
                      to load the image
                    </li>
                    <li>
                      Convert to HSV using{" "}
                      <code className="bg-white px-2 py-1 rounded">
                        cv2.cvtColor()
                      </code>{" "}
                      with COLOR_BGR2HSV
                    </li>
                    <li>
                      Calculate new size:{" "}
                      <code className="bg-white px-2 py-1 rounded">
                        (width//2, height//2)
                      </code>
                    </li>
                    <li>
                      Use{" "}
                      <code className="bg-white px-2 py-1 rounded">
                        cv2.resize()
                      </code>{" "}
                      to resize the image
                    </li>
                    <li>Display all images side by side</li>
                  </ol>
                </div>
                <CodeBlock
                  code={`import cv2

# Load image
image = cv2.imread('image.jpg')

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get original dimensions
height, width = image.shape[:2]

# Resize to half
new_size = (width // 2, height // 2)
resized_image = cv2.resize(hsv_image, new_size)

# Display results
cv2.imshow('Original', image)
cv2.imshow('HSV', hsv_image)
cv2.imshow('Resized HSV', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task2">
              <AccordionTrigger className="text-lg font-semibold">
                Task 2: Drawing Shapes on Blank Canvas
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Create a blank image and draw basic shapes like rectangles and
                  circles on it using OpenCV.
                </p>
                <CodeBlock
                  code={`import cv2
import numpy as np

# Create blank white canvas
canvas = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Draw multiple shapes
cv2.rectangle(canvas, (50, 50), (200, 150), (255, 0, 0), 3)  # Blue rectangle
cv2.circle(canvas, (400, 100), 60, (0, 255, 0), -1)  # Filled green circle
cv2.line(canvas, (100, 300), (600, 300), (0, 0, 255), 5)  # Red line
cv2.ellipse(canvas, (350, 400), (100, 50), 0, 0, 360, (255, 0, 255), -1)  # Magenta ellipse

# Add text label
cv2.putText(canvas, 'My Shapes', (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('Drawing Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task3">
              <AccordionTrigger className="text-lg font-semibold">
                Task 3: Median Blur and Center Cropping
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Load an image, first apply median blur instead of Gaussian
                  blur, then crop the center region using NumPy slicing.
                </p>
                <CodeBlock
                  code={`import cv2

# Load image
image = cv2.imread('image.jpg')

# Apply median blur (kernel size must be odd)
blurred = cv2.medianBlur(image, 5)

# Get dimensions
height, width = image.shape[:2]

# Calculate center crop coordinates (crop middle 50%)
crop_h, crop_w = height // 2, width // 2
start_y, start_x = height // 4, width // 4

# Crop center region
center_crop = blurred[start_y:start_y+crop_h, start_x:start_x+crop_w]

cv2.imshow('Original', image)
cv2.imshow('Median Blurred', blurred)
cv2.imshow('Center Crop', center_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task4">
              <AccordionTrigger className="text-lg font-semibold">
                Task 4: Multi-line Text on Images
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Load an image and add multi-line text with different font
                  styles and colors, then repeat on a blank image but position
                  text in different corners.
                </p>
                <CodeBlock
                  code={`import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')
image = cv2.resize(image, (800, 600))

# Add multi-line text with different styles
texts = [
    ("Computer Vision", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3),
    ("Lab Manual 01", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2),
    ("FAST NUCES", (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2),
]

for text, pos, font, scale, color, thickness in texts:
    cv2.putText(image, text, pos, font, scale, color, thickness)

# Create blank image
blank = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Add text to corners
corner_texts = [
    ("Top Left", (10, 30)),
    ("Top Right", (650, 30)),
    ("Bottom Left", (10, 580)),
    ("Bottom Right", (600, 580)),
]

for text, pos in corner_texts:
    cv2.putText(blank, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.imshow('Text on Image', image)
cv2.imshow('Text on Blank', blank)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task5">
              <AccordionTrigger className="text-lg font-semibold">
                Task 5: Adaptive Thresholding and Rotation
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Load a grayscale image, apply adaptive thresholding instead of
                  fixed binary threshold, then rotate the image by 45° instead
                  of 60°.
                </p>
                <CodeBlock
                  code={`import cv2

# Load grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

# Rotate 45 degrees
height, width = image.shape
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(adaptive_thresh, rotation_matrix, (width, height))

cv2.imshow('Original', image)
cv2.imshow('Adaptive Threshold', adaptive_thresh)
cv2.imshow('Rotated 45°', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task6">
              <AccordionTrigger className="text-lg font-semibold">
                Task 6: Weighted Blending with Histogram Analysis
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Load two images, resize them to same size, blend them with
                  different weights (e.g. 0.7 and 0.3), convert the result to
                  grayscale, and then plot its histogram before and after
                  equalization.
                </p>
                <CodeBlock
                  code={`import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Weighted blending
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

# Convert to grayscale
gray_blended = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized = cv2.equalizeHist(gray_blended)

# Calculate histograms
hist_before = cv2.calcHist([gray_blended], [0], None, [256], [0, 256])
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Plot histograms
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.title('Blended Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist_before)
plt.title('Histogram Before')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.plot(hist_after)
plt.title('Histogram After Equalization')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

cv2.imshow('Equalized', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="task7">
              <AccordionTrigger className="text-lg font-semibold">
                Task 7: Bitwise Operations with Shapes
              </AccordionTrigger>
              <AccordionContent className="space-y-4">
                <p className="text-slate-700">
                  Create two binary images (one with a circle, one with a
                  rectangle) and perform bitwise operations (AND, OR, XOR, NOT)
                  between them, then show the results side by side in one grid.
                </p>
                <CodeBlock
                  code={`import cv2
import numpy as np

# Create two binary images
img1 = np.zeros((400, 400), dtype=np.uint8)
img2 = np.zeros((400, 400), dtype=np.uint8)

# Draw circle on first image
cv2.circle(img1, (200, 200), 100, 255, -1)

# Draw rectangle on second image
cv2.rectangle(img2, (100, 100), (300, 300), 255, -1)

# Perform bitwise operations
bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_xor = cv2.bitwise_xor(img1, img2)
bitwise_not = cv2.bitwise_not(img1)

# Create a grid to display all results
top_row = np.hstack([img1, img2, bitwise_and])
bottom_row = np.hstack([bitwise_or, bitwise_xor, bitwise_not])
result_grid = np.vstack([top_row, bottom_row])

# Add labels (optional, using putText)
cv2.putText(result_grid, 'Circle', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(result_grid, 'Rectangle', (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(result_grid, 'AND', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(result_grid, 'OR', (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(result_grid, 'XOR', (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
cv2.putText(result_grid, 'NOT', (850, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

cv2.imshow('Bitwise Operations Grid', result_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()`}
                />
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-lg">
            <h3 className="font-semibold text-amber-900 mb-2">💡 Pro Tips</h3>
            <ul className="space-y-2 text-amber-800">
              <li>
                • Always check image dimensions before operations to avoid
                errors
              </li>
              <li>
                • Use{" "}
                <code className="bg-white px-2 py-1 rounded">
                  cv2.waitKey(0)
                </code>{" "}
                to keep windows open until a key is pressed
              </li>
              <li>• Remember OpenCV uses BGR format, not RGB</li>
              <li>
                • Use NumPy's array slicing for efficient cropping operations
              </li>
              <li>
                • Subplot functions help display multiple images in organized
                grids
              </li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
