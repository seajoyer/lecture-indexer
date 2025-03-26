#!/usr/bin/env sh

# Create a directory for the output
mkdir -p output

# Create a sample input file with YouTube educational videos
cat > lecture_urls.txt << 'EOF'
# YouTube educational videos for batch processing
# Lines starting with # are ignored

# Python Programming
https://www.youtube.com/watch?v=rfscVS0vtbw  # Learn Python - Full Course for Beginners
https://www.youtube.com/watch?v=8DvywoWv6fI  # Python for Everybody - Full University Python Course

# Mathematics
https://www.youtube.com/watch?v=WUvTyaaNkzM  # Essence of calculus
https://www.youtube.com/watch?v=fNk_zzaMoSs  # Vectors, what even are they?
EOF

# Run the batch processor with the input file
python batch_processor.py lecture_urls.txt --output-dir output

# To limit processing to just 2 videos:
# python batch_processor.py lecture_urls.txt --output-dir output --max-videos 2

# To enable debug logging:
# python batch_processor.py lecture_urls.txt --output-dir output --debug

# Make sure to set the YouTube API key as an environment variable
# export YOUTUBE_API_KEY='your-api-key-here'
