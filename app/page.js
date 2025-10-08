import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  BookOpen,
  Image,
  Palette,
  ScanEye,
  Scissors,
  Waves,
} from "lucide-react";

export default function Home() {
  const labs = [
    {
      number: "01",
      title: "Introduction to Computer Vision",
      description:
        "Basic image operations, libraries, and fundamental concepts",
      icon: BookOpen,
      href: "/lab-01",
      topics: ["Image Acquisition", "OpenCV Basics", "Image Coordinates"],
    },
    {
      number: "02",
      title: "Image Enhancement & Transformations",
      description: "Point processing, color spaces, and histogram equalization",
      icon: Palette,
      href: "/lab-02",
      topics: [
        "Brightness Adjustment",
        "Color Transformations",
        "Histogram Equalization",
      ],
    },
    {
      number: "03",
      title: "Filtering & Fourier Transforms",
      description:
        "Linear/non-linear filtering, sampling, and frequency domain",
      icon: Waves,
      href: "/lab-03",
      topics: ["Gaussian Blur", "Edge Detection", "2D Fourier Transform"],
    },
    {
      number: "04",
      title: "Feature Extraction",
      description: "HOG, LBP, edge detection, and corner detection techniques",
      icon: ScanEye,
      href: "/lab-04",
      topics: ["HOG Features", "Canny Edge Detector", "Harris Corner"],
    },
    {
      number: "05",
      title: "Image Segmentation",
      description: "Thresholding, region growing, watershed, and clustering",
      icon: Scissors,
      href: "/lab-05",
      topics: ["Thresholding", "Watershed", "K-Means Clustering"],
    },
    {
      number: "05B",
      title: "Advanced Techniques",
      description: "Wavelets, Hough transforms, and SIFT feature extraction",
      icon: Image,
      href: "/lab-05b",
      topics: ["Wavelet Transform", "Hough Lines", "SIFT Features"],
    },
  ];

  return (
    <div className="space-y-12">
      <div className="text-center space-y-4">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Computer Vision Lab Guide
        </h1>
        <p className="text-xl text-slate-600 max-w-2xl mx-auto">
          AI4002 - Fall 2025 | FAST NUCES Karachi
        </p>
        <p className="text-slate-500">
          Your comprehensive resource for computer vision concepts, code
          examples, and practical applications
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {labs.map((lab) => {
          const Icon = lab.icon;
          return (
            <Link key={lab.number} href={lab.href}>
              <Card className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1 h-full border-2 hover:border-blue-400">
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <Icon className="w-6 h-6 text-blue-600" />
                    </div>
                    <span className="text-sm font-semibold text-blue-600">
                      Lab {lab.number}
                    </span>
                  </div>
                  <CardTitle className="text-2xl">{lab.title}</CardTitle>
                  <CardDescription className="text-base">
                    {lab.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-slate-700">
                      Key Topics:
                    </p>
                    <ul className="text-sm text-slate-600 space-y-1">
                      {lab.topics.map((topic, idx) => (
                        <li key={idx} className="flex items-center gap-2">
                          <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                          {topic}
                        </li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200">
        <CardHeader>
          <CardTitle>About This Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-slate-700">
          <p>
            This comprehensive guide covers all lab manuals for the Computer
            Vision course. Each lab includes:
          </p>
          <ul className="space-y-2 ml-6">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>
                <strong>Detailed Explanations:</strong> Understand the theory
                behind each technique
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>
                <strong>Code Examples:</strong> Working Python code with OpenCV
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>
                <strong>Practical Tasks:</strong> Hands-on exercises to
                reinforce learning
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 mt-1">•</span>
              <span>
                <strong>Real-world Applications:</strong> See how concepts apply
                to actual problems
              </span>
            </li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
