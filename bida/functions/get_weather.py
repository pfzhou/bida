import json

"""The name MUST BE the SAME as the FUNCTION NAME in the JSON FILE, CASE SENSITIVE"""

def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location"""
    """当前为演示示例"""
    
    weather_info = {
        "location": location,
        "temperature": "26",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

