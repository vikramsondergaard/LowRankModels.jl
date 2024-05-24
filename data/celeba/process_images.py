from get_similarities import view_images_from_string

def main():
    images = [
        # GROUP 0
        ['(split_0_img177.png, split_0_img282.png)', '(split_0_img319.png, split_0_img386.png)', 
        '(split_0_img426.png, split_0_img51.png)', '(split_0_img482.png, split_0_img468.png)', 
        '(split_0_img251.png, split_0_img51.png)', '(split_0_img18.png, split_0_img435.png)', 
        '(split_0_img238.png, split_0_img50.png)', '(split_0_img403.png, split_0_img224.png)', 
        '(split_0_img206.png, split_0_img264.png)', '(split_0_img183.png, split_0_img53.png)'],
        # GROUP 1
        ['(split_1_img418.png, split_1_img158.png)',
        '(split_1_img508.png, split_1_img178.png)',
        '(split_1_img134.png, split_1_img268.png)',
        '(split_1_img182.png, split_1_img407.png)',
        '(split_1_img450.png, split_1_img178.png)',
        '(split_1_img532.png, split_1_img516.png)',
        '(split_1_img400.png, split_1_img41.png)',
        '(split_1_img484.png, split_1_img270.png)',
        '(split_1_img279.png, split_1_img336.png)',
        '(split_1_img78.png, split_1_img410.png)'],
        # GROUP 2
        ['(split_2_img416.png, split_2_img167.png)',
        '(split_2_img26.png, split_2_img333.png)',
        '(split_2_img302.png, split_2_img88.png)',
        '(split_2_img503.png, split_2_img195.png)',
        '(split_2_img274.png, split_2_img337.png)',
        '(split_2_img207.png, split_2_img286.png)',
        '(split_2_img444.png, split_2_img309.png)',
        '(split_2_img188.png, split_2_img254.png)',
        '(split_2_img431.png, split_2_img154.png)',
        '(split_2_img150.png, split_2_img8.png)'],
        # GROUP 3
        ['(split_3_img51.png, split_3_img26.png)',
         '(split_3_img457.png, split_3_img328.png)',
         '(split_3_img252.png, split_3_img260.png)',
         '(split_3_img54.png, split_3_img384.png)',
         '(split_3_img344.png, split_3_img169.png)',
         '(split_3_img286.png, split_3_img456.png)',
         '(split_3_img355.png, split_3_img162.png)',
         '(split_3_img250.png, split_3_img328.png)',
         '(split_3_img445.png, split_3_img201.png)',
         '(split_3_img152.png, split_3_img93.png)'],
        # GROUP 4
        ['(split_4_img37.png, split_4_img186.png)', '(split_4_img22.png, split_4_img416.png)', 
        '(split_4_img54.png, split_4_img98.png)', '(split_4_img539.png, split_4_img501.png)', 
        '(split_4_img186.png, split_4_img179.png)', '(split_4_img223.png, split_4_img503.png)', 
        '(split_4_img37.png, split_4_img179.png)', '(split_4_img482.png, split_4_img23.png)', 
        '(split_4_img383.png, split_4_img513.png)', '(split_4_img201.png, split_4_img415.png)'],
        # GROUP 5
        ['(split_5_img138.png, split_5_img146.png)', '(split_5_img213.png, split_5_img441.png)', 
        '(split_5_img85.png, split_5_img305.png)', '(split_5_img106.png, split_5_img441.png)', 
        '(split_5_img455.png, split_5_img223.png)', '(split_5_img52.png, split_5_img138.png)', 
        '(split_5_img106.png, split_5_img242.png)', '(split_5_img213.png, split_5_img106.png)', 
        '(split_5_img52.png, split_5_img146.png)', '(split_5_img85.png, split_5_img260.png)'],
        # GROUP 6
        ['(split_6_img507.png, split_6_img448.png)', '(split_6_img276.png, split_6_img307.png)', 
        '(split_6_img490.png, split_6_img300.png)', '(split_6_img88.png, split_6_img159.png)', 
        '(split_6_img535.png, split_6_img216.png)', '(split_6_img154.png, split_6_img241.png)', 
        '(split_6_img90.png, split_6_img54.png)', '(split_6_img226.png, split_6_img202.png)', 
        '(split_6_img330.png, split_6_img260.png)', '(split_6_img101.png, split_6_img4.png)'],
        # GROUP 7
        ['(split_7_img538.png, split_7_img279.png)', '(split_7_img37.png, split_7_img484.png)', 
        '(split_7_img93.png, split_7_img253.png)', '(split_7_img92.png, split_7_img87.png)', 
        '(split_7_img95.png, split_7_img533.png)', '(split_7_img366.png, split_7_img215.png)', 
        '(split_7_img47.png, split_7_img144.png)', '(split_7_img366.png, split_7_img203.png)', 
        '(split_7_img150.png, split_7_img442.png)', '(split_7_img215.png, split_7_img394.png)'],
        # GROUP 8
        ['(split_8_img405.png, split_8_img425.png)', '(split_8_img50.png, split_8_img312.png)', 
        '(split_8_img84.png, split_8_img430.png)', '(split_8_img226.png, split_8_img60.png)', 
        '(split_8_img205.png, split_8_img143.png)', '(split_8_img115.png, split_8_img335.png)', 
        '(split_8_img501.png, split_8_img291.png)', '(split_8_img477.png, split_8_img23.png)', 
        '(split_8_img466.png, split_8_img493.png)', '(split_8_img399.png, split_8_img143.png)'],
        # GROUP 9
        ['(split_9_img326.png, split_9_img112.png)', '(split_9_img420.png, split_9_img372.png)', 
        '(split_9_img130.png, split_9_img431.png)', '(split_9_img419.png, split_9_img174.png)', 
        '(split_9_img530.png, split_9_img325.png)', '(split_9_img322.png, split_9_img56.png)', 
        '(split_9_img20.png, split_9_img224.png)', '(split_9_img437.png, split_9_img71.png)', 
        '(split_9_img445.png, split_9_img358.png)', '(split_9_img109.png, split_9_img287.png)']
    ]
    for group in images:
        for image_pair in group:
            view_images_from_string(image_pair)


if __name__ == "__main__":
    main()