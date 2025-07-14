import joblib
import leap
import time
import numpy as np

model = joblib.load('sign_language_model.joblib')


def extract_features_from_frame(frame):
    print('is it extracting features')
    features = []
    
    for hand in frame.hands:
        # Basic hand features
        hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
        palm_position = hand.palm.position
        palm_velocity = hand.palm.velocity
        palm_normal = hand.palm.normal
        
        # Extract basic features for each finger
        finger_features = []
        for finger in hand.digits:
            # For simplicity, we'll just take the tip position of each finger
            tip_position = finger.distal.next_joint
            
            finger_features.extend([tip_position.x, tip_position.y, tip_position.z])
        
        # Combine hand features and finger features
        hand_features = [
            palm_position.x, palm_position.y, palm_position.z,
            palm_velocity.x, palm_velocity.y, palm_velocity.z,
            palm_normal.x, palm_normal.y, palm_normal.z
        ]
        hand_features.extend(finger_features)
        
        features.append(hand_features)
        print('Final features: ', features)
    
    return features

def preprocess_features(features):
    return

class RealTimePredictionListener(leap.Listener):
    print('its here')
    def on_tracking_event(self, event):
        if event.tracking_frame_id % 50 == 0:
            print('On frame:', event.tracking_frame_id)
            print('hands:', event.hands)
        
            # Assuming a function to extract features from a frame
            features = extract_features_from_frame(event)

            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)


            print(features)
        
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
