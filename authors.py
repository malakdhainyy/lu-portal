import time
import random
import sys
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Optional: Fix encoding on Windows
if sys.platform.startswith('win'):
    import os
    os.system('chcp 65001')

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

org_url = "https://scholar.google.com/citations?view_op=view_org&hl=en&org=9671583371665794735"

def scrape_profile(url):
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get(url)

    profile_data = {
        'name': 'Not found',
        'h_index_total': 'Not found',
        'citations_total': 'Not found',
        'total_papers': 'Not found',
        'profile_link': url
    }

    try:
        # Extract name
        profile_data['name'] = wait.until(
            EC.presence_of_element_located((By.ID, "gsc_prf_in"))
        ).text.strip()

        # Extract citations and h-index
        try:
            metrics_table = driver.find_element(By.ID, "gsc_rsb_st")
            rows = metrics_table.find_elements(By.TAG_NAME, "tr")

            if len(rows) > 1:
                citation_row = rows[1]
                citation_values = citation_row.find_elements(By.TAG_NAME, "td")
                profile_data['citations_total'] = citation_values[1].text.strip()

            if len(rows) > 2:
                h_index_row = rows[2]
                h_index_values = h_index_row.find_elements(By.TAG_NAME, "td")
                profile_data['h_index_total'] = h_index_values[1].text.strip()
        except Exception as e:
            print(f"Error extracting citation/h-index: {e}")

        # Extract total papers
        try:
            # Scroll to bottom and load all papers (click "Show more" if exists)
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                try:
                    show_more = driver.find_element(By.ID, "gsc_bpf_more")
                    if show_more.is_enabled():
                        show_more.click()
                        time.sleep(1.5)
                    else:
                        break
                except NoSuchElementException:
                    break

            publications = driver.find_elements(By.CLASS_NAME, "gsc_a_tr")
            profile_data['total_papers'] = str(len(publications))
        except Exception as e:
            print(f"Error extracting total papers: {e}")

    except Exception as e:
        print(f"Error scraping profile: {str(e)}")
    finally:
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        return profile_data

def main():
    driver.get(org_url)
    time.sleep(2)

    current_page = 0
    max_pages = 30

    # Open CSV and write header
    with open('profiles.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name', 'Profile URL', 'Total Citations', 'H-index (Total)', 'Total Papers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while current_page < max_pages:
                current_page += 1
                print(f"\n--- Processing page {current_page} ---")

                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.gs_ai")))
                profile_links = driver.find_elements(By.CSS_SELECTOR, "h3.gs_ai_name a")
                print(f"Found {len(profile_links)} profiles on this page")

                for idx, link in enumerate(profile_links, 1):
                    try:
                        profile_url = link.get_attribute("href")
                        if not profile_url:
                            continue

                        print(f"\nProcessing profile {idx}: {profile_url}")
                        data = scrape_profile(profile_url)

                        name = data['name'].encode('utf-8', errors='ignore').decode()
                        h_index = data['h_index_total']
                        citations = data['citations_total']
                        papers = data['total_papers']

                        print(f"{name} | h-index: {h_index} | Citations: {citations} | Papers: {papers}")
                        print(f"Profile URL: {profile_url}")

                        # Write row to CSV
                        writer.writerow({
                            'Name': name,
                            'Profile URL': profile_url,
                            'Total Citations': citations,
                            'H-index (Total)': h_index,
                            'Total Papers': papers
                        })

                        time.sleep(random.uniform(2, 4))

                    except Exception as e:
                        print(f"Failed to process profile: {str(e)}")
                        continue

                try:
                    next_button = wait.until(EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "button[aria-label='Next']")))

                    if 'disabled' in next_button.get_attribute("class"):
                        print("Reached last page")
                        break

                    next_button.click()
                    wait.until(EC.staleness_of(profile_links[0]))
                    time.sleep(random.uniform(3, 5))

                except Exception as e:
                    print(f"Pagination error: {str(e)}")
                    break

        finally:
            driver.quit()
            print("\nScraping complete! Data saved to profiles.csv")

if __name__ == "__main__":
    main()
