import { Outlet, NavLink, Link } from "react-router-dom";

import github from "../../assets/github.svg";

import styles from "./Layout.module.css";

import { useLogin } from "../../authConfig";

import { LoginButton } from "../../components/LoginButton";

import logo from "../../assets/flatirons.png";

const Layout = () => {
    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <Link to="/" className={styles.headerTitleContainer}>
                        {/* <h3 className={styles.headerTitle}>GPT + Enterprise data | Sample</h3> */}
                        <img src={logo} alt="Logo" className={styles.headerLogo} />
                    </Link>
                    <nav>
                        <ul className={styles.headerNavList}>
                            {/* <li>
                                <NavLink to="/" className={({ isActive }) => (isActive ? styles.headerNavPageLinkActive : styles.headerNavPageLink)}>
                                    Chat
                                </NavLink>
                            </li>
                            <li className={styles.headerNavLeftMargin}>
                                <NavLink to="/qa" className={({ isActive }) => (isActive ? styles.headerNavPageLinkActive : styles.headerNavPageLink)}>
                                    Ask a question
                                </NavLink>
                            </li>
                            <li className={styles.headerNavLeftMargin}>
                                <a href="https://aka.ms/entgptsearch" target={"_blank"} title="Github repository link">
                                    <img
                                        src={github}
                                        alt="Github logo"
                                        aria-label="Link to github repository"
                                        width="20px"
                                        height="20px"
                                        role="img"
                                        className={styles.githubLogo}
                                    />
                                </a>
                            </li> */}
                            <li className={styles.headerNavLeftMargin}>
                                <a href="https://flatironsai.com" target="_blank" rel="noopener noreferrer">
                                    <button className={styles.navButton}>Learn About Copilot Enterprise</button>
                                </a>
                            </li>
                            <li className={styles.headerNavLeftMargin}>
                                <a href="https://flatironsai.com" target="_blank" rel="noopener noreferrer">
                                    <button className={styles.navButton}>Support and Learning</button>
                                </a>
                            </li>
                        </ul>
                    </nav>
                    {useLogin && <LoginButton />}
                </div>
            </header>

            <Outlet />
            <footer className={styles.footer} role={"contentinfo"}>
                <a href="https://flatironsai.com" target="_blank" rel="noopener norefererrer">
                    <button className={styles.footerButton}>Feedback?</button>
                </a>
            </footer>
        </div>
    );
};

export default Layout;
