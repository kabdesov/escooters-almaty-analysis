import pandas as pd
import numpy as np

def generate_mock_data():
    samples = [
        "E-scooters are blocking the sidewalks!", 
        "Great green mobility solution for Almaty.",
        "New traffic regulations are needed.", 
        "Accidents involving scooters are rising.",
        "Police are issuing fines for sidewalk riding.", 
        "Scooters are the best last-mile solution."
    ]
    for platform in ['instagram', 'facebook']:
        df = pd.DataFrame({
            'text': [np.random.choice(samples) for _ in range(30)],
            'id': range(30)
        })
        df.to_csv(f"{platform}.csv", index=False)
        print(f"Generated {platform}.csv with synthetic data for demo.")

if __name__ == "__main__":
    generate_mock_data()