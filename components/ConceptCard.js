import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function ConceptCard({
  title,
  description,
  children,
  icon: Icon,
}) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        {Icon && (
          <div className="mb-2">
            <div className="p-2 bg-blue-100 rounded-lg w-fit">
              <Icon className="w-5 h-5 text-blue-600" />
            </div>
          </div>
        )}
        <CardTitle className="text-xl">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      {children && <CardContent>{children}</CardContent>}
    </Card>
  );
}
