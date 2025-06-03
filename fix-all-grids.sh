#!/bin/bash

echo "🔧 Fixing all Material-UI Grid components in frontend..."

# Find all TypeScript/JavaScript files in components and app directories
find ./components ./app -name "*.tsx" -o -name "*.ts" | while read file; do
    if grep -q "Grid" "$file"; then
        echo "📁 Processing: $file"
        
        # Remove Grid from imports
        sed -i '' 's/,\s*Grid,/,/g' "$file"
        sed -i '' 's/Grid,\s*//' "$file"
        sed -i '' 's/,\s*Grid\s*}/}/' "$file"
        
        # Replace Grid container with Box
        sed -i '' 's/<Grid container spacing={\([0-9]*\)}>/<Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: \1 }}>/g' "$file"
        
        # Replace Grid items with Box
        sed -i '' 's/<Grid item xs={\([0-9]*\)}[^>]*>/<Box>/g' "$file"
        sed -i '' 's/<Grid item[^>]*>/<Box>/g' "$file"
        
        # Replace closing Grid tags
        sed -i '' 's/<\/Grid>/<\/Box>/g' "$file"
        
        echo "  ✅ Fixed Grid components in $file"
    fi
done

echo "🏁 Grid component fix complete!"
echo "💡 Run 'npm run build' to test the fixes"
