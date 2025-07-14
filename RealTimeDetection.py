import joblib
import leap
import time
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler


model = joblib.load('sign_language_model2.joblib')
#scaler = joblib.load('scaler.joblib') 

finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
bone_names = ['metacarpal', 'proximal', 'intermediate', 'distal']
positions = ['start', 'end', 'direction']
labels = ['x', 'y', 'z']
positions_names = {'start': 'next_joint', 'end': 'prev_joint', 'direction': 'rotation'}
sizes = ['length', 'width']
# scaler = MinMaxScaler()

#print(dir(model))

def calculate_length(prev_joint, next_joint):
    x_diff = next_joint.x - prev_joint.x
    y_diff = next_joint.y - prev_joint.y
    z_diff = next_joint.z - prev_joint.z

    length = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

    return length




def extract_features_from_frame(frame):
    #print('is it extracting features')
    features_dict = {}

    left_hand_present = False
    right_hand_present = False


    
    for hand in frame.hands:
        # Basic hand features
        prefix = "left_" if str(hand.type) == "HandType.Left" else "right_"

        if hand.type == "left_": 
            left_hand_present = True
        elif hand.type == "right_": 
            right_hand_present = True

        if not left_hand_present:
            for key in generate_feature_keys("left_"):
                features_dict[key] = 0
        
        if not right_hand_present:
            for key in generate_feature_keys("right_"):
                features_dict[key] = 0


        if not left_hand_present and not right_hand_present:
                for key in generate_feature_keys("right_"):
                    features_dict[key] = 0
                for key in generate_feature_keys("left_"):
                    features_dict[key] = 0


        

        features_dict[prefix + 'palm_position_x'] = hand.palm.position.x
        features_dict[prefix + 'palm_position_y'] = hand.palm.position.y
        features_dict[prefix + 'palm_position_z'] = hand.palm.position.z

        features_dict[prefix + 'palm_normal_x'] = hand.palm.normal.x
        features_dict[prefix + 'palm_normal_y'] = hand.palm.normal.y
        features_dict[prefix + 'palm_normal_z'] = hand.palm.normal.z

        features_dict[prefix + 'hand_direction_x'] = hand.palm.direction.x
        features_dict[prefix + 'hand_direction_y'] = hand.palm.direction.y
        features_dict[prefix + 'hand_direction_z'] = hand.palm.direction.z

        features_dict[prefix + 'palm_velocity_x'] = hand.palm.velocity.x
        features_dict[prefix + 'palm_velocity_y'] = hand.palm.velocity.y
        features_dict[prefix + 'palm_velocity_z'] = hand.palm.velocity.z


        features_dict[prefix + 'arm_direction_x'] = hand.arm.rotation.x
        features_dict[prefix + 'arm_direction_y'] = hand.arm.rotation.y
        features_dict[prefix + 'arm_direction_z'] = hand.arm.rotation.z

        features_dict[prefix + 'wrist_position_x'] = hand.arm.next_joint.x
        features_dict[prefix + 'wrist_position_y'] = hand.arm.next_joint.y
        features_dict[prefix + 'wrist_position_z'] = hand.arm.next_joint.z

        features_dict[prefix + 'elbow_position_x'] = hand.arm.prev_joint.x
        features_dict[prefix + 'elbow_position_y'] = hand.arm.prev_joint.y
        features_dict[prefix + 'elbow_position_z'] = hand.arm.prev_joint.z

        # pitch =  math.atan2(hand.palm.direction.y, -hand.palm.direction.z) 
        # roll = math.atan2(hand.palm.normal.x, -hand.palm.normal.y)
        # yaw = math.atan2(hand.palm.direction.x, -hand.palm.direction.z)


        # features_dict[prefix + 'hand_pitch'] = math.degrees(pitch)
        # features_dict[prefix + 'hand_roll'] = math.degrees(roll)
        # features_dict[prefix + 'hand_yaw'] = math.degrees(yaw)


        

        # for finger_idx, finger in enumerate(finger_names):            
        #          key = f"{prefix}{finger}_width"
        #          features_dict[key] = hand.digits[finger_idx].intermediate.width


        # for finger_idx, finger in enumerate(finger_names):
        #     key = f"{prefix}{finger}_width"
        #     prev_joint = hand.digits[finger_idx].proximal.prev_joint

        #     next_joint = hand.digits[finger_idx].distal.next_joint

        #     length =  calculate_length(prev_joint, next_joint)

        #     features_dict[key] = length

       


        for finger_idx, finger in enumerate(finger_names):            
            for bone in bone_names:
                for position in positions:
                    # Correctly use the `position` variable to get the name from `positions_names`
                    position_name = positions_names[position] 
                    for label in labels:
                        # Construct the key for the features_dict
                        key = f"{prefix}{finger}_{bone}_{position}_{label}"

                        # Dynamically get the bone object from the finger object
                        bone_object = getattr(hand.digits[finger_idx], bone)
                        
                        # Depending on the position, we might need to access different properties or methods
                        if position in ['start', 'end']:
                            # For 'start' and 'end', `position_name` points to 'next_joint' or 'prev_joint', which are properties
                            joint_object = getattr(bone_object, position_name)
                            # Now, access the label (x, y, z) from this joint object
                            value = getattr(joint_object, label)
                        elif position == 'direction':
                            # For 'direction', `position_name` is 'rotation', assuming it's a directly accessible property with sub-properties
                            direction_object = getattr(bone_object, position_name)
                            value = getattr(direction_object, label)
                        features_dict[key] = value


        #print(features_df)
                
        features_df = pd.DataFrame(features_dict, index=[0])
        # scaled_features_df = scaler.fit_transform(features_df)
        #scaled_features = scaler.fit_transform(features_df)  # Use transform, not fit_transform


        #print(features_df)
            
        
        #print('Final features: ', features_dict)
    
        return features_df


def generate_feature_keys(prefix):
    """
    Dynamically generates the keys for all features for a given hand prefix.
    Adjust this function based on the features you are actually extracting.
    """
    keys = []
    features = ['palm_position_x', 'palm_position_y', 'palm_position_z',
                'palm_normal_x', 'palm_normal_y', 'palm_normal_z',
                'hand_direction_x', 'hand_direction_y', 'hand_direction_z',
                'palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z',
                # 'hand_pitch', 'hand_roll', 'hand_yaw',
                'arm_direction_x', 'arm_direction_y', 'arm_direction_z',
                'wrist_position_x', 'wrist_position_y', 'wrist_position_z',
                'elbow_position_x', 'elbow_position_y', 'elbow_position_z',

                ]
    
    for finger in finger_names:
        # for size in sizes:
        #     key1 = f"{finger}_{size}"
        #     features.append(key1)            
        for bone in bone_names:
            for position in positions:
                for label in labels:
                    # Construct the key for the features_dict
                    key2 = f"{finger}_{bone}_{position}_{label}"
                    features.append(key2)


    for feature in features:
        keys.append(prefix + feature)
    # Add logic for generating finger bone feature keys if necessary
    return keys


class RealTimePredictionListener(leap.Listener):
    def __init__(self, prediction_callback=None):
        self.prediction_callback = prediction_callback
        #print('its here')
    def on_tracking_event(self, event):
        if event.tracking_frame_id % 50 == 0:
            #print('On frame:', event.tracking_frame_id)
            #print('hands:', event.hands)
        
            # Assuming a function to extract features from a frame
            features = extract_features_from_frame(event)
            #print(features)

            prediction = model.predict(features)

            if self.prediction_callback:
                self.prediction_callback(prediction)

        # Use the prediction in your application
            print(f"Predicted gesture: {prediction}")        

def main():
    # Setup Leap Motion controller and listener
    listener = RealTimePredictionListener()
    connection = leap.Connection()
    
    # Attach the listener
    connection.add_listener(listener)
    
    with connection.open():
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
