def soil_tool(ph):

    if ph < 6:
        return "Soil is acidic. Add lime."

    elif ph > 7.5:
        return "Soil is alkaline. Add compost."

    else:
        return "Soil pH is optimal."
