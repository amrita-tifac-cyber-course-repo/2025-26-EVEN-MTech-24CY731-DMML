from src.predict import predict_system

print("=== Predictive Maintenance System ===")

# Take input from user
Type = int(input("Enter Type (0=L, 1=M, 2=H): "))
air_temp = float(input("Enter Air Temperature: "))
process_temp = float(input("Enter Process Temperature: "))
speed = float(input("Enter Rotational Speed: "))
torque = float(input("Enter Torque: "))
tool_wear = float(input("Enter Tool Wear: "))

features = [Type, air_temp, process_temp, speed, torque, tool_wear]

prob, decision = predict_system(features)

print("\n===== RESULT =====")
print("Failure Probability:", prob)
print("Recommended Action:", decision)