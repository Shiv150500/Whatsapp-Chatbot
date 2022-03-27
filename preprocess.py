from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def scrape_courses(course): 
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--disable-gpu") 
    options.add_argument("window-size=1024,768") 
    options.add_argument("--no-sandbox")
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, l ike Gecko) Chrome/60.0.3112.50 Safari/537.36'
    options.add_argument('user-agent={0}'.format(user_agent))
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options
    )

    url = "https://www.udemy.com/courses/search/?src=ukw&q=" 
    m_course = course.replace(' ', '+')
    f_url = url + m_course
    driver.get(f_url) 
    links = []
    msg = ""

    body = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.
    CLASS_NAME, "course-list--container--3zXPS")))
    cards = body.find_elements_by_class_name("popper--popper--19faV.popper-- popper-hover--4YJ5J")
    for i in range(len(cards)):
        attr = cards[i].find_element_by_tag_name("a") 
        link = attr.get_attribute("href") 
        links.append(link)
    driver.quit()
    msg += "Here are the top " + str(len(cards)) + " " + course +" courses:" 
    for j in range(len(links)):
        msg += "\n" + links[j]
    return(msg)

