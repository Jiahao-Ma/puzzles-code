### Command-line Usage for Image-to-Clips
Some rules to follow when using the command line for Image-to-Clips:
- **Short or long clips**: Modify `--NPuzzles` to change the number of generated clips, affecting their duration.
- **Overlap**: Adjusting the overlap between patches by setting `--min_overlap` and `--max_overlap` can help create more diverse video clips.
- **Zoom in or out**: Use `--min_box_size_scale` and `--max_box_size_scale` to control the size of the patches, allowing for zooming in and out.
- **Camera rotation**: Enable the `--Augment` flag to apply rotation to a subset of patches, diversifying viewpoints.

```bash
python Image2Clips.py --input examples/I2C/alien.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 8 --min_box_size_scale 0.4 --max_box_size_scale 0.8 \
                        --min_overlap 0.1 --max_overlap 0.3 
                        
python Image2Clips.py --input examples/I2C/lamp.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 --Augment \
                        --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                        --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                        
python Image2Clips.py --input examples/I2C/boys.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 --Augment \
                        --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                        --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                        
python Image2Clips.py --input examples/I2C/living_room.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 10 --min_box_size_scale 0.2 --max_box_size_scale 0.6 \
                        --min_overlap 0.1 --max_overlap 0.3 
                        
python Image2Clips.py --input examples/I2C/mountain2.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 10 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 
                        
python Image2Clips.py --input examples/I2C/rabbit.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 6 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 --Augment \
                        --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                        --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
                        
python Image2Clips.py --input examples/I2C/studio_room.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 8 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 --Augment \
                        --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                        --max_rotation_angle 30.0 --front_back_ratio_thresh 0.6
                        
python Image2Clips.py --input examples/I2C/wall.jpg --output examples/I2C --threshold 0.01 \
                        --NPuzzles 8 --min_box_size_scale 0.2 --max_box_size_scale 0.5 \
                        --min_overlap 0.1 --max_overlap 0.3 --Augment \
                        --rotate_ratio 0.2 --rotate_min_coverage 0.2 --rotate_max_coverage 0.8 \
                        --max_rotation_angle 60.0 --front_back_ratio_thresh 0.6
```