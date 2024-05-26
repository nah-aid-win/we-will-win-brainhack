from typing import Dict


class NLPManager:
    arr = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    
    def __init__(self):
        # initialize the model here
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.tool_answerer = pipeline("question-answering", model="toolmodel",device=device)
        self.target_answerer = pipeline("question-answering", model="targetmodel",device=device)
        self.heading_answerer = pipeline("question-answering", model="headingmodel",device=device)
        
    def fun(self, s):
        s = s.lower().split()
        ans = ""
        for i in range(len(s)):
            for j in range(10):
                if(s[i].count(self.arr[j])>0):
                    ans += str(j)
                    break
            if(len(ans)==3):
                break
        return ans

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        
        question = "What is the tool to use to destroy target"
        tool = self.tool_answerer(question=question, context=context)["answer"]
        if(tool != "EMP"):
            tool = tool.lower()
        if(tool == "machine guns"):
            tool = "machine gun"


        question = "what is the target to destroy"
        target = self.target_answerer(question=question, context=context)["answer"].lower()
        if(target[-1] == 's'):
            target = target[0:-1]



        question = "What is the heading"
        heading = self.heading_answerer(question=question, context=context)["answer"]
        heading = self.fun(heading)
        
        return {"heading": heading, "tool": tool, "target": target}

